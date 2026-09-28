package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Expr

/** Common serial traversal contract for library-built staged ranges. */
abstract class GpuTraversal[T] private[dsl] ():
  def foreach(body: Expr[T] => (BlockBuilder ?=> Unit))(using BlockBuilder, DslSourcePosition): Unit

  final def flatMap[U](expand: Expr[T] => GpuTraversal[U]): FlatMappedGpuRange[U] =
    new FlatMappedGpuRange(body =>
      foreach { value =>
        val inner = ExpressionStaging.expression(expand(value))
        inner.foreach(body)
      }
    )
