package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

/** A symbolic half-open, unit-stride range executed serially by each CUDA thread. */
final class GpuRange private[dsl] (
    val indexName: String,
    val from: Expr[Int],
    val until: Expr[Int]
):
  def map[T](valueAt: Expr[Int] => Expr[T]): MappedGpuRange[T] =
    new MappedGpuRange(this, valueAt)

  def foreach(
      body: Expr[Int] => (BlockBuilder ?=> Unit)
  )(using builder: BlockBuilder, position: DslSourcePosition): Unit =
    gpuFor(indexName, from, until)(body)

/** Composed expression builders, not a materialized JVM or GPU collection.
  * Callbacks run once per terminal during staging and should only build expressions.
  * Reusing an expression does not memoize its device-side value.
  */
final class MappedGpuRange[T] private[dsl] (
    private val range: GpuRange,
    private val valueAt: Expr[Int] => Expr[T]
):
  def map[U](transform: Expr[T] => Expr[U]): MappedGpuRange[U] =
    new MappedGpuRange(range, valueAt.andThen(transform))

  def sum[A](
      initial: Expr[A],
      policy: ReductionPolicy = ReductionPolicy.Strict
  )(using
      rule: AccumulatorType[T, A],
      addition: AdditiveType[A],
      position: DslSourcePosition
  ): Expr[A] =
    reduceSum(range.indexName, range.from, range.until, initial, policy)(valueAt)

  def foreach(
      body: Expr[T] => (BlockBuilder ?=> Unit)
  )(using builder: BlockBuilder, position: DslSourcePosition): Unit =
    range.foreach(index => body(valueAt(index)))
