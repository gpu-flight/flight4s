package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, Load}
import flight4s.core.types.CudaType

/** A serial traversal of flat expression tuples; no device tuple storage is allocated. */
final class GpuTupleTraversal[Values <: NonEmptyTuple] private[dsl] (
    private val stage: (Values => (BlockBuilder ?=> Unit)) => ((BlockBuilder, DslSourcePosition) ?=> Unit)
) extends GpuValueTraversal[Values]:
  def foreach(body: Values => (BlockBuilder ?=> Unit))(using BlockBuilder, DslSourcePosition): Unit =
    stage(body)

  def map[U](transform: Values => Expr[U]): FlatMappedGpuRange[U] =
    new FlatMappedGpuRange(body => foreach(value => body(ExpressionStaging.expression(transform(value)))))

  def map[Next <: NonEmptyTuple](transform: Values => Next)(using TupleFoldState[Next]): GpuTupleTraversal[Next] =
    new GpuTupleTraversal(body => foreach(value => body(ExpressionStaging.expression(transform(value)))))

  def filter(predicate: Values => Expr[Boolean])(using position: DslSourcePosition): GpuTupleTraversal[Values] =
    new GpuTupleTraversal(body => foreach { value =>
      when(ExpressionStaging.expression(predicate(value))) { body(value) }(using summon[BlockBuilder], position)
    })

  def withFilter(predicate: Values => Expr[Boolean])(using DslSourcePosition): GpuTupleTraversal[Values] =
    filter(predicate)

  def flatMap[U](expand: Values => GpuTraversal[U]): FlatMappedGpuRange[U] =
    new FlatMappedGpuRange(body => foreach { value =>
      val inner = ExpressionStaging.expression(expand(value))
      inner.foreach(body)
    })

  def flatMap[Next <: NonEmptyTuple](expand: Values => GpuTupleTraversal[Next])(
      using TupleFoldState[Next]
  ): GpuTupleTraversal[Next] =
    new GpuTupleTraversal(body => foreach { value =>
      val inner = ExpressionStaging.expression(expand(value))
      inner.foreach(body)
    })

  def foldLeft[A](accumulatorName: String, initial: Expr[A])(
      step: (Expr[A], Values) => Expr[A]
  )(using CudaType[A], BlockBuilder, DslSourcePosition): Expr[A] =
    val accumulator = local(accumulatorName, initial)
    foreach(value => accumulator := ExpressionStaging.expression(step(accumulator.read, value)))
    Load(accumulator, summon[DslSourcePosition].span)
