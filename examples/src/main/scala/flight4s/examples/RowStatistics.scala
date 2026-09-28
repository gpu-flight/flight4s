package flight4s.examples

import scala.util.Using

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Expr
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, CudaDriverException, CudaDriverFailure, NvrtcCompiler}

object RowStatistics:
  final case class Result(means: Array[Double], populationVariances: Array[Double])

  val definition = kernel("rowStatistics", params(
    input[Float]("values"), output[Double]("means"), output[Double]("variances"),
    value[Int]("rows"), value[Int]("columns")
  )) { bindings =>
    val (values, means, variances, rows, columns) = bindings
    val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
    when(row < rows) {
      val (count, mean, m2) = gpuRange("column", literal(0), columns)
        .map(column => convert.f32ToF64(values(row * columns + column).read))
        .foldLeft("moments", (literal(0), literal(0.0), literal(0.0)))(update)
      means(row) := mean
      variances(row) := m2 / convert.i32ToF64(count)
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

  /** One serial Welford fold per finite Float row; Double population statistics. */
  def run(values: Array[Float], rows: Int, columns: Int, deviceOrdinal: Int = 0): Result =
    validateInput(values, rows, columns)
    if rows == 0 then Result(Array.emptyDoubleArray, Array.emptyDoubleArray)
    else
      val opened = CudaContext.open(deviceOrdinal).fold(failure => throw CudaDriverException(failure), identity)
      Using.resource(opened) { context =>
        val result = for
          artifact <- NvrtcCompiler.compile(generatedModule, context.computeCapability, "row_statistics.cu")
            .left.map(failure => s"${failure.message}\n${failure.compileLog}")
          module <- driver(context.load(artifact))
          function <- driver(module.function(generated))
          stream <- driver(context.createStream())
          input <- driver(context.allocate[Float](values.length))
          means <- driver(context.allocate[Double](rows))
          variances <- driver(context.allocate[Double](rows))
          _ <- driver(input.copyFrom(values))
          _ <- function.launch(
            definition.bind((input, means, variances, rows, columns)),
            LaunchConfig(Grid.x(blockCount(rows)), LaunchBlock.x(128)), stream
          ).left.map(_.message)
          _ <- driver(stream.synchronize())
          outputMeans <- driver(means.copyToArray())
          outputVariances <- driver(variances.copyToArray())
        yield Result(outputMeans, outputVariances)
        result.fold(message => throw IllegalStateException(message), identity)
      }

  def main(args: Array[String]): Unit =
    args.toList match
      case List("--cuda-source") => println(generated.cudaSource)
      case Nil =>
        val values = Array(1.0f, 2.0f, 3.0f, 4.0f,
          1000000.0f, 1000001.0f, 1000002.0f, 1000003.0f,
          -7.0f, -7.0f, -7.0f, -7.0f)
        val result = run(values, rows = 3, columns = 4)
        result.means.indices.foreach { row =>
          println(f"row $row: mean=${result.means(row)}%.7f populationVariance=${result.populationVariances(row)}%.7f")
        }
      case _ => throw IllegalArgumentException("usage: RowStatistics [--cuda-source]")

  private[examples] def validateInput(values: Array[Float], rows: Int, columns: Int): Unit =
    require(rows >= 0, "rows must be non-negative")
    require(columns > 0, "columns must be positive")
    require(rows.toLong * columns == values.length.toLong, "row-major shape must match input length")
    require(values.forall(java.lang.Float.isFinite), "statistics example requires finite values")

  private[examples] def blockCount(rows: Int): Int =
    ((rows.toLong + 127L) / 128L).toInt

  private def driver[A](result: Either[CudaDriverFailure, A]): Either[String, A] =
    result.left.map(failure => s"${failure.message}\n${failure.infoLog}\n${failure.errorLog}".trim)
