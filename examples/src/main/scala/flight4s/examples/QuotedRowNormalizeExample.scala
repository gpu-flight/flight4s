package flight4s.examples

import scala.annotation.experimental
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.frontend.ScalaKernel.*
import flight4s.runtime.cuda.*

/** End-to-end quoted kernel: block reductions, device math, a conversion and a
  * thread-local cache normalize each row to zero mean and unit variance. */
object QuotedRowNormalizeExample:
  @experimental
  def rowNormalize = kernel("quotedRowNormalize", params(input[Float]("data"), output[Float]("out"),
      value[Int]("columns"))) { p =>
    val scratch = blockReduction[Float](64)
    val cache = localArray[Float](8)
    val row = blockIdx.x
    val lane = threadIdx.x
    var partial = 0.0f
    for k <- deviceRange(0, 8) do
      val column = lane + k * 64
      cache(k) = if column < p._3 then p._1(row * p._3 + column) else 0.0f
      partial += cache(k)
    val total = scratch.sum(partial)
    val mean = total / p._3.toFloat
    var squares = 0.0f
    for k <- deviceRange(0, 8) do
      val column = lane + k * 64
      if column < p._3 then
        val centered = cache(k) - mean
        squares += centered * centered
    val squareTotal = scratch.sum(squares)
    val scale = rsqrt(squareTotal / p._3.toFloat + 0.00001f)
    for k <- deviceRange(0, 8) do
      val column = lane + k * 64
      if column < p._3 then p._2(row * p._3 + column) = (cache(k) - mean) * scale
  }.requiringBlock(LaunchBlock.x(64))

  @experimental
  def main(arguments: Array[String]): Unit =
    val rows = 128
    val columns = 300
    val data = Array.tabulate(rows * columns) { index =>
      val row = index / columns
      val column = index % columns
      math.sin(row * 0.7 + column * 0.13).toFloat * 2.0f + row * 0.01f
    }
    val definition = rowNormalize
    def unwrap[A](result: Either[?, A], step: String): A = result match
      case Right(value) => value
      case Left(failure) => sys.error(s"$step failed: $failure")
    val context = unwrap(CudaContext.open(0), "opening CUDA device 0")
    try
      println(s"compute capability: ${context.computeCapability}")
      val generated = unwrap(CudaCodegen.generate(definition), "generating CUDA source")
      val artifact = unwrap(NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource,
        generated.sourceMap, generated.compilerOptions, Vector(generated)),
        context.computeCapability, "quoted_row_normalize.cu"), "NVRTC compilation")
      val module = unwrap(context.load(artifact), "loading the module")
      try
        val function = unwrap(module.function(generated), "binding the kernel function")
        val dataBuffer = unwrap(context.allocate[Float](rows * columns), "allocating the input")
        val outBuffer = unwrap(context.allocate[Float](rows * columns), "allocating the output")
        try
          unwrap(dataBuffer.copyFrom(data), "uploading the input")
          unwrap(function.launch(definition.bind((dataBuffer, outBuffer, columns)),
            LaunchConfig(Grid.x(rows), LaunchBlock.x(64))), "launching")
          unwrap(context.synchronize(), "synchronizing")
          val produced = unwrap(outBuffer.copyToArray(), "downloading the output")

          var worstDifference = 0.0
          var worstMean = 0.0
          var worstVariance = 0.0
          for row <- 0 until rows do
            val slice = data.slice(row * columns, row * columns + columns)
            val mean = slice.foldLeft(0.0)(_ + _) / columns
            val variance = slice.foldLeft(0.0)((sum, v) => sum + (v - mean) * (v - mean)) / columns
            val scale = 1.0 / math.sqrt(variance + 0.00001f)
            var producedSum = 0.0
            var producedSquares = 0.0
            for column <- 0 until columns do
              val index = row * columns + column
              val reference = (slice(column) - mean) * scale
              worstDifference = math.max(worstDifference, math.abs(produced(index) - reference))
              producedSum += produced(index)
              producedSquares += produced(index).toDouble * produced(index)
            worstMean = math.max(worstMean, math.abs(producedSum / columns))
            worstVariance = math.max(worstVariance, math.abs(producedSquares / columns - 1.0))
          println(f"max |gpu - cpu reference|:   $worstDifference%.3e")
          println(f"max |row mean|:              $worstMean%.3e")
          println(f"max |row variance - 1|:      $worstVariance%.3e")
          require(worstDifference < 1e-3, "GPU normalization diverged from the CPU reference")
          require(worstMean < 1e-4 && worstVariance < 1e-3, "normalized rows are off target")
          println(s"normalized $rows rows of $columns columns on the GPU")
        finally
          outBuffer.close()
          dataBuffer.close()
      finally module.close()
    finally
      context.close()
      println("closed the context and released GPU resources")
