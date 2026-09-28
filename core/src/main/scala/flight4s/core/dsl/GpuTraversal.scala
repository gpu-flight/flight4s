package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, SourceSpan}

/** Common serial traversal contract for library-built staged ranges. */
abstract class GpuTraversal[T] private[dsl] ():
  def foreach(body: Expr[T] => (BlockBuilder ?=> Unit))(using BlockBuilder, DslSourcePosition): Unit

  /** Tuple components advance simultaneously from the previous iteration. */
  final def foldLeft[State <: NonEmptyTuple](stateName: String, initial: State)(
      step: (State, Expr[T]) => State
  )(using state: TupleFoldState[State], builder: BlockBuilder, position: DslSourcePosition): State =
    val locals = state.declare(stateName, initial, 0)
    foreach { value =>
      val next = ExpressionStaging.expression(step(locals.read(SourceSpan.Unknown), value))
      // Snapshot every next component before changing any old component.
      val snapshots = state.snapshot(stateName, next, 0)
      locals.assign(snapshots)
    }
    locals.read(position.span)

  final def flatMap[U](expand: Expr[T] => GpuTraversal[U]): FlatMappedGpuRange[U] =
    new FlatMappedGpuRange(body =>
      foreach { value =>
        val inner = ExpressionStaging.expression(expand(value))
        inner.foreach(body)
      }
    )
