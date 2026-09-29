package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock}
import flight4s.core.types.{AdditiveType, CudaType}

final class BlockReduction[T] private[dsl] (
    scratch: SharedArray[T, Rank1],
    shape: LaunchBlock,
    threadCount: Int,
    valueType: CudaType[T]
):
  def sum(value: Expr[T])(using
      AdditiveType[T], BlockBuilder, DslSourcePosition
  ): Expr[T] = reduceTree(value)(_ + _)

  def reduceTree(value: Expr[T])(combine: (Expr[T], Expr[T]) => Expr[T])(using
      builder: BlockBuilder, position: DslSourcePosition
  ): Expr[T] = reduceTree(builder.freshName("reduction"), value)(combine)

  def sum(name: String, value: Expr[T])(using
      AdditiveType[T], BlockBuilder, DslSourcePosition
  ): Expr[T] = reduceTree(name, value)(_ + _)

  def reduceTree(name: String, value: Expr[T])(combine: (Expr[T], Expr[T]) => Expr[T])(using
      builder: BlockBuilder, position: DslSourcePosition
  ): Expr[T] =
    given CudaType[T] = valueType
    builder.requireBlock(shape, position.span)
    val result = local(name, value)
    scoped {
      val rank = let(s"${name}_rank", threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
      scratch(rank) := result.read
      barrier()
      var distance = 1
      while distance < threadCount do
        val stride = distance * 2
        when((rank % literal(stride)) === literal(0)) {
          val left = scratch(rank).read
          val right = scratch(rank + literal(distance)).read
          scratch(rank) := ExpressionStaging.expression(combine(left, right))
        }
        barrier()
        distance = stride
      result := scratch(literal(0)).read
      // Every reader must finish before another call can overwrite the shared scratch.
      barrier()
    }
    result.read
