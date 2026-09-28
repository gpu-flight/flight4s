package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, Load}
import flight4s.core.types.CudaType

/** A serial traversal of named expression fields; no device product storage is allocated. */
final class GpuProductTraversal[Value <: Product] private[dsl] (
    private val stage: (Value => (BlockBuilder ?=> Unit)) => ((BlockBuilder, DslSourcePosition) ?=> Unit)
) extends GpuValueTraversal[Value]:
  def foreach(body: Value => (BlockBuilder ?=> Unit))(using BlockBuilder, DslSourcePosition): Unit =
    stage(body)

  def map[U](transform: Value => Expr[U]): FlatMappedGpuRange[U] =
    new FlatMappedGpuRange(body => foreach(value => body(ExpressionStaging.expression(transform(value)))))

  def map[Next <: NonEmptyTuple](transform: Value => Next)(using TupleFoldState[Next]): GpuTupleTraversal[Next] =
    new GpuTupleTraversal(body => foreach(value => body(ExpressionStaging.expression(transform(value)))))

  def filter(predicate: Value => Expr[Boolean])(using position: DslSourcePosition): GpuProductTraversal[Value] =
    new GpuProductTraversal(body => foreach { value =>
      when(ExpressionStaging.expression(predicate(value))) { body(value) }(using summon[BlockBuilder], position)
    })

  def withFilter(predicate: Value => Expr[Boolean])(using DslSourcePosition): GpuProductTraversal[Value] =
    filter(predicate)

  def flatMap[U](expand: Value => GpuTraversal[U]): FlatMappedGpuRange[U] =
    new FlatMappedGpuRange(body => foreach { value =>
      val inner = ExpressionStaging.expression(expand(value))
      inner.foreach(body)
    })

  def flatMap[Next <: NonEmptyTuple](expand: Value => GpuTupleTraversal[Next])(
      using TupleFoldState[Next]
  ): GpuTupleTraversal[Next] =
    new GpuTupleTraversal(body => foreach { value =>
      val inner = ExpressionStaging.expression(expand(value))
      inner.foreach(body)
    })

  def foldLeft[A](accumulatorName: String, initial: Expr[A])(
      step: (Expr[A], Value) => Expr[A]
  )(using CudaType[A], BlockBuilder, DslSourcePosition): Expr[A] =
    val accumulator = local(accumulatorName, initial)
    foreach(value => accumulator := ExpressionStaging.expression(step(accumulator.read, value)))
    Load(accumulator, summon[DslSourcePosition].span)
