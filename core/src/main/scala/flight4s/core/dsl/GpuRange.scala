package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

/** A symbolic half-open range executed serially by each CUDA thread. */
final class GpuRange private[dsl] (
    val indexName: String,
    val from: Expr[Int],
    val until: Expr[Int],
    val step: Int = 1
) extends GpuTraversal[Int]:
  def by(step: Int): GpuRange =
    new GpuRange(indexName, from, until, step)

  def map[T](valueAt: Expr[Int] => Expr[T]): MappedGpuRange[T] =
    new MappedGpuRange(this, valueAt)

  def filter(predicate: Expr[Int] => Expr[Boolean])(using DslSourcePosition): FilteredGpuRange[Int] =
    map(identity).filter(predicate)

  def withFilter(predicate: Expr[Int] => Expr[Boolean])(using DslSourcePosition): FilteredGpuRange[Int] =
    filter(predicate)

  /** Stages one ordered local update per element and returns its read-only result expression. */
  def foldLeft[A](accumulatorName: String, initial: Expr[A])(
      step: (Expr[A], Expr[Int]) => Expr[A]
  )(using valueType: CudaType[A], builder: BlockBuilder, position: DslSourcePosition): Expr[A] =
    val accumulator = local(accumulatorName, initial)
    foreach { index => accumulator := ExpressionStaging.expression(step(accumulator.read, index)) }
    Load(accumulator, position.span)

  /** Pair components advance simultaneously from the previous iteration's state. */
  def foldLeft[A, B](stateName: String, initial: (Expr[A], Expr[B]))(
      step: ((Expr[A], Expr[B]), Expr[Int]) => (Expr[A], Expr[B])
  )(using CudaType[A], CudaType[B], BlockBuilder, DslSourcePosition): (Expr[A], Expr[B]) =
    PairFold.stage(stateName, initial)(step)(body => foreach(body))

  def foreach(
      body: Expr[Int] => (BlockBuilder ?=> Unit)
  )(using builder: BlockBuilder, position: DslSourcePosition): Unit =
    gpuFor(indexName, from, until, step)(body)

/** Composed expression builders, not a materialized JVM or GPU collection.
  * Callbacks run once per terminal during staging and should only build expressions.
  * Reusing an expression does not memoize its device-side value.
  */
final class MappedGpuRange[T] private[dsl] (
    private val range: GpuRange,
    private val valueAt: Expr[Int] => Expr[T]
) extends GpuTraversal[T]:
  def map[U](transform: Expr[T] => Expr[U]): MappedGpuRange[U] =
    new MappedGpuRange(range, valueAt.andThen(transform))

  def filter(predicate: Expr[T] => Expr[Boolean])(using DslSourcePosition): FilteredGpuRange[T] =
    new FilteredGpuRange(range, (index, body) =>
      val value = ExpressionStaging.expression(valueAt(index))
      when(ExpressionStaging.expression(predicate(value))) { body(value) }
    )

  def withFilter(predicate: Expr[T] => Expr[Boolean])(using DslSourcePosition): FilteredGpuRange[T] =
    filter(predicate)

  def foldLeft[A](accumulatorName: String, initial: Expr[A])(
      step: (Expr[A], Expr[T]) => Expr[A]
  )(using valueType: CudaType[A], builder: BlockBuilder, position: DslSourcePosition): Expr[A] =
    range.foldLeft(accumulatorName, initial)((accumulator, index) => step(accumulator, valueAt(index)))

  def foldLeft[A, B](stateName: String, initial: (Expr[A], Expr[B]))(
      step: ((Expr[A], Expr[B]), Expr[T]) => (Expr[A], Expr[B])
  )(using CudaType[A], CudaType[B], BlockBuilder, DslSourcePosition): (Expr[A], Expr[B]) =
    range.foldLeft(stateName, initial)((state, index) => step(state, valueAt(index)))

  def sum[A](
      initial: Expr[A],
      policy: ReductionPolicy = ReductionPolicy.Strict
  )(using
      rule: AccumulatorType[T, A],
      addition: AdditiveType[A],
      position: DslSourcePosition
  ): Expr[A] =
    reduceSum(range.indexName, range.from, range.until, initial, policy, range.step)(valueAt)

  def foreach(
      body: Expr[T] => (BlockBuilder ?=> Unit)
  )(using builder: BlockBuilder, position: DslSourcePosition): Unit =
    range.foreach(index => body(ExpressionStaging.expression(valueAt(index))))
