package flight4s.examples

import scala.util.Using

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, Rank1, SharedArray}
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, CudaDriverException, CudaDriverFailure, NvrtcCompiler}

object BlockRowStatistics:
  val ThreadsPerBlock: Int = 128

  val definition = kernel("blockRowStatistics", params(
    input[Float]("values"), output[Double]("means"), output[Double]("variances"),
    value[Int]("rows"), value[Int]("columns")
  )) { bindings =>
    val (values, means, variances, rows, columns) = bindings
    val counts = sharedArray[Int]("counts", ThreadsPerBlock)
    val partialMeans = sharedArray[Double]("partialMeans", ThreadsPerBlock)
    val partialM2 = sharedArray[Double]("partialM2", ThreadsPerBlock)
    val row = let("row", blockIdx.x)
    when(row < rows) {
      val (count, mean, m2) = gpuRange("column", threadIdx.x, columns).by(ThreadsPerBlock)
        .map(column => convert.f32ToF64(values(row * columns + column).read))
        .foldLeft("moments", (literal(0), literal(0.0), literal(0.0)))(update)
      counts(threadIdx.x) := count
      partialMeans(threadIdx.x) := mean
      partialM2(threadIdx.x) := m2
      mergePartials(counts, partialMeans, partialM2)
      when(threadIdx.x === literal(0)) {
        means(row) := partialMeans(literal(0)).read
        variances(row) := partialM2(literal(0)).read / convert.i32ToF64(counts(literal(0)).read)
      }
    }
  }.requiringBlock(LaunchBlock.x(ThreadsPerBlock))

  private[examples] def update(state: (Expr[Int], Expr[Double], Expr[Double]), x: Expr[Double]) =
    val (count, mean, m2) = state
    val nextCount = count + literal(1)
    val delta = x - mean
    val nextMean = mean + delta / convert.i32ToF64(nextCount)
    (nextCount, nextMean, m2 + delta * (x - nextMean))

  private[examples] def mergePartials(counts: SharedArray[Int, Rank1], means: SharedArray[Double, Rank1],
      m2: SharedArray[Double, Rank1])(using BlockBuilder): Unit =
    barrier()
    Iterator.iterate(ThreadsPerBlock / 2)(_ / 2).takeWhile(_ > 0).foreach { stride =>
      when(threadIdx.x < literal(stride)) {
        val right = threadIdx.x + literal(stride)
        val rightCount = let("rightCount", counts(right).read)
        when(rightCount > literal(0)) {
          val leftCount = let("leftCount", counts(threadIdx.x).read)
          val combinedCount = let("combinedCount", leftCount + rightCount)
          val leftMean = let("leftMean", means(threadIdx.x).read)
          val rightMean = let("rightMean", means(right).read)
          val delta = let("delta", rightMean - leftMean)
          val rightWeight = let("rightWeight", convert.i32ToF64(rightCount) / convert.i32ToF64(combinedCount))
          val nextMean = let("nextMean", choose(leftCount === literal(0))(rightMean)(leftMean + delta * rightWeight))
          val nextM2 = let("nextM2", choose(leftCount === literal(0))(m2(right).read)(
            m2(threadIdx.x).read + m2(right).read + delta * delta * convert.i32ToF64(leftCount) * rightWeight))
          // Finish every shared-memory read before overwriting the left state.
          counts(threadIdx.x) := combinedCount
          means(threadIdx.x) := nextMean
          m2(threadIdx.x) := nextM2
        }
      }
      barrier()
    }

  val generated = CudaCodegen.generate(definition)
    .fold(error => throw IllegalStateException(error.message), identity)
  private val generatedModule = GeneratedCudaModule(
    generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
  )

  /** Finite Float rows, Double population statistics, one 128-thread block per row. */
  def run(values: Array[Float], rows: Int, columns: Int, deviceOrdinal: Int = 0): RowStatistics.Result =
    RowStatistics.validateInput(values, rows, columns)
    if rows == 0 then RowStatistics.Result(Array.emptyDoubleArray, Array.emptyDoubleArray)
    else
      val opened = CudaContext.open(deviceOrdinal).fold(failure => throw CudaDriverException(failure), identity)
      Using.resource(opened) { context =>
        val result = for
          artifact <- NvrtcCompiler.compile(generatedModule, context.computeCapability, "block_row_statistics.cu")
            .left.map(failure => s"${failure.message}\n${failure.compileLog}")
          module <- driver(context.load(artifact))
          function <- driver(module.function(generated))
          stream <- driver(context.createStream())
          input <- driver(context.allocate[Float](values.length))
          means <- driver(context.allocate[Double](rows))
          variances <- driver(context.allocate[Double](rows))
          _ <- driver(input.copyFrom(values))
          _ <- function.launch(definition.bind((input, means, variances, rows, columns)), launchConfig(rows), stream)
            .left.map(_.message)
          _ <- driver(stream.synchronize())
          outputMeans <- driver(means.copyToArray())
          outputVariances <- driver(variances.copyToArray())
        yield RowStatistics.Result(outputMeans, outputVariances)
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
        val values = Array.tabulate(rows * columns) { i =>
          if i / columns == 2 then -7.0f
          else (i % columns).toFloat + (i / columns) * 1000000.0f
        }
        val result = run(values, rows, columns)
        result.means.indices.foreach { row =>
          println(f"row $row: mean=${result.means(row)}%.7f populationVariance=${result.populationVariances(row)}%.7f")
        }
      case _ => throw IllegalArgumentException("usage: BlockRowStatistics [--cuda-source]")

  private def driver[A](result: Either[CudaDriverFailure, A]): Either[String, A] =
    result.left.map(failure => s"${failure.message}\n${failure.infoLog}\n${failure.errorLog}".trim)
