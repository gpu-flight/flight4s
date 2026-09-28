package flight4s.examples

import scala.util.Using

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, CudaDriverException, CudaDriverFailure, NvrtcCompiler}

object RowSoftmax:
  val definition = kernel("rowSoftmax", params(
    input[Float]("logits"), output[Float]("probabilities"),
    value[Int]("rows"), value[Int]("columns")
  )) { bindings =>
    val (logits, probabilities, rows, columns) = bindings
    val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
    when(row < rows) {
      val elements = gpuRange("column", literal(0), columns)
        .map(column => logits(row * columns + column).read)
      val maximum = elements.foldLeft("maximum", literal(Float.NegativeInfinity)) {
        (best, x) => choose(x > best)(x)(best)
      }
      val denominator = let("denominator", elements.map(x => exp(x - maximum)).sum(literal(0.0f)))
      gpuRange("writeColumn", literal(0), columns).foreach { column =>
        val index = row * columns + column
        probabilities(index) := exp(logits(index).read - maximum) / denominator
      }
    }
  }

  val generated = CudaCodegen.generate(definition)
    .fold(error => throw IllegalStateException(error.message), identity)
  private val generatedModule = GeneratedCudaModule(
    generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
  )

  /** Finite row-major Float logits; one serial row per CUDA thread. */
  def run(values: Array[Float], rows: Int, columns: Int, deviceOrdinal: Int = 0): Array[Float] =
    validateInput(values, rows, columns)
    if rows == 0 then Array.emptyFloatArray
    else
      val opened = CudaContext.open(deviceOrdinal).fold(failure => throw CudaDriverException(failure), identity)
      Using.resource(opened) { context =>
        val result = for
          artifact <- NvrtcCompiler.compile(generatedModule, context.computeCapability, "row_softmax.cu")
            .left.map(failure => s"${failure.message}\n${failure.compileLog}")
          module <- driver(context.load(artifact))
          function <- driver(module.function(generated))
          stream <- driver(context.createStream())
          logits <- driver(context.allocate[Float](values.length))
          probabilities <- driver(context.allocate[Float](values.length))
          _ <- driver(logits.copyFrom(values))
          _ <- function.launch(
            definition.bind((logits, probabilities, rows, columns)),
            LaunchConfig(Grid.x(blockCount(rows)), LaunchBlock.x(128)),
            stream
          ).left.map(_.message)
          _ <- driver(stream.synchronize())
          output <- driver(probabilities.copyToArray())
        yield output
        result.fold(message => throw IllegalStateException(message), identity)
      }

  def main(args: Array[String]): Unit =
    args.toList match
      case List("--cuda-source") => println(generated.cudaSource)
      case Nil =>
        val rows = 3
        val columns = 4
        val logits = Array(1000.0f, 1001.0f, 1002.0f, 1003.0f,
          -1000.0f, -999.0f, -998.0f, -997.0f, 0.0f, 0.0f, 0.0f, 0.0f)
        val probabilities = run(logits, rows, columns)
        probabilities.grouped(columns).zipWithIndex.foreach { (row, index) =>
          println(s"row $index: ${row.map(value => f"$value%.7f").mkString(" ")} (sum=${row.sum})")
        }
      case _ => throw IllegalArgumentException("usage: RowSoftmax [--cuda-source]")

  private[examples] def validateInput(values: Array[Float], rows: Int, columns: Int): Unit =
    require(rows >= 0, "rows must be non-negative")
    require(columns > 0, "columns must be positive")
    require(rows.toLong * columns == values.length.toLong, "row-major shape must match input length")
    require(values.forall(java.lang.Float.isFinite), "softmax example requires finite logits")

  private[examples] def blockCount(rows: Int): Int =
    ((rows.toLong + 127L) / 128L).toInt

  private def driver[A](result: Either[CudaDriverFailure, A]): Either[String, A] =
    result.left.map(failure => s"${failure.message}\n${failure.infoLog}\n${failure.errorLog}".trim)
