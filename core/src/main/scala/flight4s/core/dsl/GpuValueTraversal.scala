package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, SourceSpan}
import flight4s.core.types.CudaType

/** Shared ordered composition for scalar, tuple, and named-product expression traversals. */
abstract class GpuValueTraversal[Value] private[dsl] ():
  def foreach(body: Value => (BlockBuilder ?=> Unit))(using BlockBuilder, DslSourcePosition): Unit

  def foldLeft[A](accumulatorName: String, initial: Expr[A])(
      step: (Expr[A], Value) => Expr[A]
  )(using CudaType[A], BlockBuilder, DslSourcePosition): Expr[A]

  final def foldLeft[A](initial: Expr[A])(step: (Expr[A], Value) => Expr[A])(using
      valueType: CudaType[A], builder: BlockBuilder, position: DslSourcePosition
  ): Expr[A] = foldLeft(builder.freshName("fold"), initial)(step)

  final def foldLeft[State <: Product](initial: State)(step: (State, Value) => State)(using
      state: ProductFoldState[State], builder: BlockBuilder, position: DslSourcePosition
  ): State = foldLeft(builder.freshName("fold"), initial)(step)

  final def foldLeft[State <: NonEmptyTuple](initial: State)(step: (State, Value) => State)(using
      state: TupleFoldState[State], builder: BlockBuilder, position: DslSourcePosition
  ): State = foldLeft(builder.freshName("fold"), initial)(step)

  final def map[Next <: Product](transform: Value => Next)(using ProductFoldState[Next]): GpuProductTraversal[Next] =
    new GpuProductTraversal(body => foreach(value => body(ExpressionStaging.expression(transform(value)))))

  final def flatMap[Next <: Product](expand: Value => GpuProductTraversal[Next])(
      using ProductFoldState[Next]
  ): GpuProductTraversal[Next] =
    new GpuProductTraversal(body => foreach { value =>
      val inner = ExpressionStaging.expression(expand(value))
      inner.foreach(body)
    })

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
