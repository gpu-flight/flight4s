package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.SourceSpan

/** Shared ordered fold semantics for scalar-expression and tuple-expression traversals. */
abstract class GpuValueTraversal[Value] private[dsl] ():
  def foreach(body: Value => (BlockBuilder ?=> Unit))(using BlockBuilder, DslSourcePosition): Unit

  /** Named case-class fields advance simultaneously, just like tuple components. */
  final def foldLeft[State <: Product](stateName: String, initial: State)(
      step: (State, Value) => State
  )(using state: ProductFoldState[State], builder: BlockBuilder, position: DslSourcePosition): State =
    val locals = state.declare(stateName, initial)
    foreach { value =>
      val next = ExpressionStaging.expression(step(locals.read(SourceSpan.Unknown), value))
      val snapshots = state.snapshot(stateName, next)
      locals.assign(snapshots)
    }
    locals.read(position.span)

  /** Tuple components advance simultaneously from the previous iteration. */
  final def foldLeft[State <: NonEmptyTuple](stateName: String, initial: State)(
      step: (State, Value) => State
  )(using state: TupleFoldState[State], builder: BlockBuilder, position: DslSourcePosition): State =
    val locals = state.declare(stateName, initial, 0)
    foreach { value =>
      val next = ExpressionStaging.expression(step(locals.read(SourceSpan.Unknown), value))
      // Snapshot every next component before changing any old component.
      val snapshots = state.snapshot(stateName, next, 0)
      locals.assign(snapshots)
    }
    locals.read(position.span)
