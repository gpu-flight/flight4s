package flight4s.examples

import scala.util.Using

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, CudaDriverException, CudaDriverFailure, NvrtcCompiler}

object BlockRowLayerNorm:
  val ThreadsPerBlock: Int = BlockRowStatistics.ThreadsPerBlock

  val definition = kernel("blockRowLayerNorm", paramsTuple((
    input[Float]("values"), input[Float]("gain"), input[Float]("bias"), output[Float]("out"),
    value[Int]("rows"), value[Int]("columns"), value[Double]("epsilon")
  ))) { bindings =>
    val (values, gain, bias, out, rows, columns, epsilon) = bindings
    val counts = sharedArray[Int]("counts", ThreadsPerBlock)
    val partialMeans = sharedArray[Double]("partialMeans", ThreadsPerBlock)
    val partialM2 = sharedArray[Double]("partialM2", ThreadsPerBlock)
    val row = let("row", blockIdx.x)
    when(row < rows) {
      val (count, mean, m2) = gpuRange("column", threadIdx.x, columns).by(ThreadsPerBlock)
        .map(column => convert.f32ToF64(values(row * columns + column).read))
        .foldLeft("moments", (literal(0), literal(0.0), literal(0.0)))(BlockRowStatistics.update)
      counts(threadIdx.x) := count
      partialMeans(threadIdx.x) := mean
      partialM2(threadIdx.x) := m2
      // The final merge barrier publishes lane zero's statistics to the entire block.
      BlockRowStatistics.mergePartials(counts, partialMeans, partialM2)
      val rowMean = let("rowMean", partialMeans(literal(0)).read)
      val denominator = let("denominator", sqrt(
        partialM2(literal(0)).read / convert.i32ToF64(counts(literal(0)).read) + epsilon))
      gpuRange("outputColumn", threadIdx.x, columns).by(ThreadsPerBlock).foreach { column =>
        val index = row * columns + column
        val normalized = (convert.f32ToF64(values(index).read) - rowMean) / denominator
        out(index) := convert.f64ToF32(
          normalized * convert.f32ToF64(gain(column).read) + convert.f32ToF64(bias(column).read))
      }
    }
  }.requiringBlock(LaunchBlock.x(ThreadsPerBlock))

  val generated = CudaCodegen.generate(definition)
    .fold(error => throw IllegalStateException(error.message), identity)
  private val generatedModule = GeneratedCudaModule(
    generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))

  /** One 128-thread block per row; same host and numerical contract as RowLayerNorm. */
  def run(values: Array[Float], gain: Array[Float], bias: Array[Float], rows: Int, columns: Int,
      epsilon: Double = 1e-5, deviceOrdinal: Int = 0): Array[Float] =
    RowLayerNorm.validateInput(values, gain, bias, rows, columns, epsilon)
    if rows == 0 then Array.emptyFloatArray
    else
      val opened = CudaContext.open(deviceOrdinal).fold(failure => throw CudaDriverException(failure), identity)
      Using.resource(opened) { context =>
        val result = for
          artifact <- NvrtcCompiler.compile(generatedModule, context.computeCapability, "block_row_layer_norm.cu")
            .left.map(failure => s"${failure.message}\n${failure.compileLog}")
          module <- driver(context.load(artifact))
          function <- driver(module.function(generated))
          stream <- driver(context.createStream())
          input <- driver(context.allocate[Float](values.length))
          gainBuffer <- driver(context.allocate[Float](columns))
          biasBuffer <- driver(context.allocate[Float](columns))
          out <- driver(context.allocate[Float](values.length))
          _ <- driver(input.copyFrom(values))
          _ <- driver(gainBuffer.copyFrom(gain))
          _ <- driver(biasBuffer.copyFrom(bias))
          _ <- function.launch(
            definition.bind((input, gainBuffer, biasBuffer, out, rows, columns, epsilon)),
            launchConfig(rows), stream).left.map(_.message)
          _ <- driver(stream.synchronize())
          output <- driver(out.copyToArray())
        yield output
        result.fold(message => throw IllegalStateException(message), identity)
      }

  private[examples] def launchConfig(rows: Int): LaunchConfig =
    LaunchConfig(Grid.x(rows), LaunchBlock.x(ThreadsPerBlock))

  def main(args: Array[String]): Unit =
    args.toList match
      case List("--cuda-source") => println(generated.cudaSource)
      case Nil =>
        val result = run(Array(1.0f, 3.0f, 1000001.0f, 1000003.0f, -7.0f, -7.0f),
          Array(2.0f, -4.0f), Array(0.25f, 1.0f), 3, 2, epsilon = 3.0)
        result.grouped(2).zipWithIndex.foreach { (row, index) =>
          println(s"row $index: ${row.map(x => f"$x%.7f").mkString(" ")}")
        }
      case _ => throw IllegalArgumentException("usage: BlockRowLayerNorm [--cuda-source]")

  private def driver[A](result: Either[CudaDriverFailure, A]): Either[String, A] =
    result.left.map(failure => s"${failure.message}\n${failure.infoLog}\n${failure.errorLog}".trim)
