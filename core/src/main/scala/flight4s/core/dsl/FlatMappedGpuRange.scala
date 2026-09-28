package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.CudaType

/** Nested serial iteration without materializing an intermediate device collection. */
final class FlatMappedGpuRange[T] private[dsl] (
    private val stage: (Expr[T] => (BlockBuilder ?=> Unit)) => ((BlockBuilder, DslSourcePosition) ?=> Unit)
) extends GpuTraversal[T]:
  def foreach(body: Expr[T] => (BlockBuilder ?=> Unit))(using BlockBuilder, DslSourcePosition): Unit =
    stage(body)

  def map[U](transform: Expr[T] => Expr[U]): FlatMappedGpuRange[U] =
    new FlatMappedGpuRange(body => foreach(value => body(ExpressionStaging.expression(transform(value)))))

  def filter(predicate: Expr[T] => Expr[Boolean])(using position: DslSourcePosition): FlatMappedGpuRange[T] =
    new FlatMappedGpuRange(body => foreach { value =>
      when(ExpressionStaging.expression(predicate(value))) { body(value) }(using summon[BlockBuilder], position)
    })

  def withFilter(predicate: Expr[T] => Expr[Boolean])(using DslSourcePosition): FlatMappedGpuRange[T] =
    filter(predicate)

  def foldLeft[A](accumulatorName: String, initial: Expr[A])(
      step: (Expr[A], Expr[T]) => Expr[A]
  )(using CudaType[A], BlockBuilder, DslSourcePosition): Expr[A] =
    val accumulator = local(accumulatorName, initial)
    foreach(value => accumulator := ExpressionStaging.expression(step(accumulator.read, value)))
    Load(accumulator, summon[DslSourcePosition].span)

  def foldLeft[A, B](stateName: String, initial: (Expr[A], Expr[B]))(
      step: ((Expr[A], Expr[B]), Expr[T]) => (Expr[A], Expr[B])
  )(using CudaType[A], CudaType[B], BlockBuilder, DslSourcePosition): (Expr[A], Expr[B]) =
    PairFold.stage(stateName, initial)(step)(body => foreach(body))
