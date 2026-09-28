package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Expr

/** Common serial traversal contract for library-built staged ranges. */
abstract class GpuTraversal[T] private[dsl] () extends GpuValueTraversal[Expr[T]]:
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
