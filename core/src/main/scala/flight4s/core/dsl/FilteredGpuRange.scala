package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.CudaType

/** Serial staged iteration with ordered GPU guards, not a compacted collection. */
final class FilteredGpuRange[T] private[dsl] (
    private val range: GpuRange,
    // A continuation keeps each later map, predicate, and terminal inside preceding guards.
    private val stageElement: (Expr[Int], Expr[T] => (BlockBuilder ?=> Unit)) => (BlockBuilder ?=> Unit)
):
  def map[U](transform: Expr[T] => Expr[U]): FilteredGpuRange[U] =
    new FilteredGpuRange(range, (index, body) =>
      stageElement(index, value => body(ExpressionStaging.expression(transform(value))))
    )

  def filter(predicate: Expr[T] => Expr[Boolean])(using DslSourcePosition): FilteredGpuRange[T] =
    new FilteredGpuRange(range, (index, body) =>
      stageElement(index, value =>
        when(ExpressionStaging.expression(predicate(value))) { body(value) }
      )
    )

  def withFilter(predicate: Expr[T] => Expr[Boolean])(using DslSourcePosition): FilteredGpuRange[T] =
    filter(predicate)

  def foreach(body: Expr[T] => (BlockBuilder ?=> Unit))(using
      builder: BlockBuilder,
      position: DslSourcePosition
  ): Unit =
    range.foreach(index => stageElement(index, body))

  def foldLeft[A](accumulatorName: String, initial: Expr[A])(
      step: (Expr[A], Expr[T]) => Expr[A]
  )(using valueType: CudaType[A], builder: BlockBuilder, position: DslSourcePosition): Expr[A] =
    val accumulator = local(accumulatorName, initial)
    foreach(value => accumulator := ExpressionStaging.expression(step(accumulator.read, value)))
    Load(accumulator, position.span)
