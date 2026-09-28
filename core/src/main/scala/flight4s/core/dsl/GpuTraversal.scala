package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, Load}
import flight4s.core.types.{AccumulatorType, AdditiveType}

/** Common serial traversal contract for library-built staged ranges. */
abstract class GpuTraversal[T] private[dsl] () extends GpuValueTraversal[Expr[T]]:
  /** Named ordered accumulation, including guarded/nested traversals and explicit promotion. */
  final def sum[A](accumulatorName: String, initial: Expr[A])(using
      rule: AccumulatorType[T, A],
      addition: AdditiveType[A],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): Expr[A] =
    val accumulator = local(accumulatorName, initial)
    foreach(value => accumulator := accumulator.read + value.toAccumulator[A])
    Load(accumulator, position.span)

  final def map[Values <: NonEmptyTuple](transform: Expr[T] => Values)(using
      TupleFoldState[Values]
  ): GpuTupleTraversal[Values] =
    new GpuTupleTraversal(body => foreach(value => body(ExpressionStaging.expression(transform(value)))))

  final def flatMap[Values <: NonEmptyTuple](expand: Expr[T] => GpuTupleTraversal[Values])(using
      TupleFoldState[Values]
  ): GpuTupleTraversal[Values] =
    new GpuTupleTraversal(body => foreach { value =>
      val inner = ExpressionStaging.expression(expand(value))
      inner.foreach(body)
    })

  final def flatMap[U](expand: Expr[T] => GpuTraversal[U]): FlatMappedGpuRange[U] =
    new FlatMappedGpuRange(body =>
      foreach { value =>
        val inner = ExpressionStaging.expression(expand(value))
        inner.foreach(body)
      }
    )
