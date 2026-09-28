package flight4s.examples

import scala.util.Using

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Expr
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, CudaDriverException, CudaDriverFailure, NvrtcCompiler}

object RowLayerNorm:
  val definition = kernel("rowLayerNorm", paramsTuple((
    input[Float]("values"), input[Float]("gain"), input[Float]("bias"), output[Float]("out"),
    value[Int]("rows"), value[Int]("columns"), value[Double]("epsilon")
  ))) { bindings =>
    val (values, gain, bias, out, rows, columns, epsilon) = bindings
    val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
    when(row < rows) {
      val (count, mean, m2) = gpuRange("column", literal(0), columns)
        .map(column => convert.f32ToF64(values(row * columns + column).read))
        .foldLeft("moments", (literal(0), literal(0.0), literal(0.0)))(update)
      val denominator = let("denominator", sqrt(m2 / convert.i32ToF64(count) + epsilon))
      gpuRange("outputColumn", literal(0), columns).foreach { column =>
        val index = row * columns + column
        val normalized = (convert.f32ToF64(values(index).read) - mean) / denominator
        out(index) := convert.f64ToF32(
          normalized * convert.f32ToF64(gain(column).read) + convert.f32ToF64(bias(column).read))
      }
    }
  }

  private def update(state: (Expr[Int], Expr[Double], Expr[Double]), x: Expr[Double]) =
    val (count, mean, m2) = state
    val nextCount = count + literal(1)
    val delta = x - mean
    val nextMean = mean + delta / convert.i32ToF64(nextCount)
    (nextCount, nextMean, m2 + delta * (x - nextMean))

  val generated = CudaCodegen.generate(definition)
    .fold(error => throw IllegalStateException(error.message), identity)
  private val generatedModule = GeneratedCudaModule(
    generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
  )

  /** Row-wise population normalization with Double arithmetic and Float affine output. */
  def run(values: Array[Float], gain: Array[Float], bias: Array[Float], rows: Int, columns: Int,
      epsilon: Double = 1e-5, deviceOrdinal: Int = 0): Array[Float] =
    validateInput(values, gain, bias, rows, columns, epsilon)
    if rows == 0 then Array.emptyFloatArray
    else
      val opened = CudaContext.open(deviceOrdinal).fold(failure => throw CudaDriverException(failure), identity)
      Using.resource(opened) { context =>
        val result = for
          artifact <- NvrtcCompiler.compile(generatedModule, context.computeCapability, "row_layer_norm.cu")
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
            LaunchConfig(Grid.x(RowStatistics.blockCount(rows)), LaunchBlock.x(128)), stream
          ).left.map(_.message)
          _ <- driver(stream.synchronize())
          output <- driver(out.copyToArray())
        yield output
        result.fold(message => throw IllegalStateException(message), identity)
      }

  private[examples] def validateInput(values: Array[Float], gain: Array[Float], bias: Array[Float],
      rows: Int, columns: Int, epsilon: Double): Unit =
    RowStatistics.validateInput(values, rows, columns)
    require(gain.length == columns && bias.length == columns, "gain and bias must match the column count")
    require(gain.forall(java.lang.Float.isFinite) && bias.forall(java.lang.Float.isFinite),
      "layer normalization requires finite gain and bias")
    require(java.lang.Double.isFinite(epsilon) && epsilon > 0.0, "epsilon must be finite and positive")

  def main(args: Array[String]): Unit =
    args.toList match
      case List("--cuda-source") => println(generated.cudaSource)
      case Nil =>
        val values = Array(1.0f, 2.0f, 3.0f, 4.0f,
          1000001.0f, 1000002.0f, 1000003.0f, 1000004.0f,
          -7.0f, -7.0f, -7.0f, -7.0f)
        val result = run(values, Array(1.0f, 2.0f, -1.0f, 0.0f),
          Array(0.0f, 0.5f, 1.0f, -2.0f), rows = 3, columns = 4)
        result.grouped(4).zipWithIndex.foreach { (row, index) =>
          println(s"row $index: ${row.map(x => f"$x%.7f").mkString(" ")}")
        }
      case _ => throw IllegalArgumentException("usage: RowLayerNorm [--cuda-source]")

  private def driver[A](result: Either[CudaDriverFailure, A]): Either[String, A] =
    result.left.map(failure => s"${failure.message}\n${failure.infoLog}\n${failure.errorLog}".trim)
