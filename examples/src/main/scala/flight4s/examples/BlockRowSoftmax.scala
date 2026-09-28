package flight4s.examples

import scala.util.Using

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, Rank1, SharedArray}
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, CudaDriverException, CudaDriverFailure, NvrtcCompiler}

object BlockRowSoftmax:
  val ThreadsPerBlock: Int = 128

  val definition = kernel("blockRowSoftmax", params(
    input[Float]("logits"), output[Float]("probabilities"),
    value[Int]("rows"), value[Int]("columns")
  )) { bindings =>
    val (logits, probabilities, rows, columns) = bindings
    val scratch = sharedArray[Float]("scratch", ThreadsPerBlock)
    val row = let("row", blockIdx.x)
    when(row < rows) {
      val elements = gpuRange("column", threadIdx.x, columns).by(ThreadsPerBlock)
        .map(column => logits(row * columns + column).read)
      val partialMaximum = elements.foldLeft("partialMaximum", literal(Float.NegativeInfinity)) {
        (best, x) => choose(x > best)(x)(best)
      }
      scratch(threadIdx.x) := partialMaximum
      reduceScratch(scratch)((left, right) => choose(right > left)(right)(left))
      val maximum = let("maximum", scratch(literal(0)).read)
      // Every thread must snapshot the maximum before scratch is reused for sums.
      barrier()

      scratch(threadIdx.x) := elements.map(x => exp(x - maximum)).sum(literal(0.0f))
      reduceScratch(scratch)(_ + _)
      val denominator = let("denominator", scratch(literal(0)).read)
      gpuRange("writeColumn", threadIdx.x, columns).by(ThreadsPerBlock).foreach { column =>
        val index = row * columns + column
        probabilities(index) := exp(logits(index).read - maximum) / denominator
      }
    }
  }.requiringBlock(LaunchBlock.x(ThreadsPerBlock))

  val generated = CudaCodegen.generate(definition)
    .fold(error => throw IllegalStateException(error.message), identity)
  private val generatedModule = GeneratedCudaModule(
    generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
  )

  private def reduceScratch(scratch: SharedArray[Float, Rank1])(
      combine: (Expr[Float], Expr[Float]) => Expr[Float]
  )(using BlockBuilder): Unit =
    barrier()
    Iterator.iterate(ThreadsPerBlock / 2)(_ / 2).takeWhile(_ > 0).foreach { stride =>
      when(threadIdx.x < literal(stride)) {
        scratch(threadIdx.x) := combine(scratch(threadIdx.x).read, scratch(threadIdx.x + literal(stride)).read)
      }
      barrier()
    }

  /** Finite row-major Float logits; one 128-thread block cooperates on each row. */
  def run(values: Array[Float], rows: Int, columns: Int, deviceOrdinal: Int = 0): Array[Float] =
    RowSoftmax.validateInput(values, rows, columns)
    if rows == 0 then Array.emptyFloatArray
    else
      val opened = CudaContext.open(deviceOrdinal).fold(failure => throw CudaDriverException(failure), identity)
      Using.resource(opened) { context =>
        val result = for
          artifact <- NvrtcCompiler.compile(generatedModule, context.computeCapability, "block_row_softmax.cu")
            .left.map(failure => s"${failure.message}\n${failure.compileLog}")
          module <- driver(context.load(artifact))
          function <- driver(module.function(generated))
          stream <- driver(context.createStream())
          logits <- driver(context.allocate[Float](values.length))
          probabilities <- driver(context.allocate[Float](values.length))
          _ <- driver(logits.copyFrom(values))
          _ <- function.launch(
            definition.bind((logits, probabilities, rows, columns)), launchConfig(rows), stream
          ).left.map(_.message)
          _ <- driver(stream.synchronize())
          output <- driver(probabilities.copyToArray())
        yield output
        result.fold(message => throw IllegalStateException(message), identity)
      }

  private[examples] def launchConfig(rows: Int): LaunchConfig =
    LaunchConfig(Grid.x(rows), LaunchBlock.x(ThreadsPerBlock))

  def main(args: Array[String]): Unit =
    args.toList match
      case List("--cuda-source") => println(generated.cudaSource)
      case Nil =>
        val rows = 3
        val columns = 257
        val logits = Array.tabulate(rows * columns) { i =>
          val row = i / columns
          if row == 2 then 0.0f
          else (i % columns - 128) / 16.0f + row * 1000.0f
        }
        val probabilities = run(logits, rows, columns)
        probabilities.grouped(columns).zipWithIndex.foreach { (row, index) =>
          println(s"row $index: ${row.take(4).map(value => f"$value%.7g").mkString(" ")} ... (sum=${row.map(_.toDouble).sum})")
        }
      case _ => throw IllegalArgumentException("usage: BlockRowSoftmax [--cuda-source]")

  private def driver[A](result: Either[CudaDriverFailure, A]): Either[String, A] =
    result.left.map(failure => s"${failure.message}\n${failure.infoLog}\n${failure.errorLog}".trim)
