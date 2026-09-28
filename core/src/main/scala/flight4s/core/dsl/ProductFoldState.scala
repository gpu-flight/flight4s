package flight4s.core.dsl

import scala.deriving.Mirror
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.SourceSpan

/** Derivable evidence for a nonempty case class whose fields are device expressions. */
sealed trait ProductFoldState[State <: Product]:
  private[dsl] def declare(name: String, initial: State)(using
      BlockBuilder, DslSourcePosition
  ): ProductFoldLocals[State]

  private[dsl] def snapshot(name: String, next: State)(using
      BlockBuilder, DslSourcePosition
  ): State

object ProductFoldState:
  def derived[State <: Product](using mirror: Mirror.ProductOf[State])(
      using fields: TupleFoldState[mirror.MirroredElemTypes],
      nonEmpty: mirror.MirroredElemTypes <:< NonEmptyTuple
  ): ProductFoldState[State] = new ProductFoldState[State]:
    private[dsl] def declare(name: String, initial: State)(using
        BlockBuilder, DslSourcePosition
    ): ProductFoldLocals[State] =
      val locals = fields.declare(name, Tuple.fromProductTyped(initial), 0)
      new ProductFoldLocals[State]:
        def read(span: SourceSpan): State =
          ExpressionStaging.expression(mirror.fromProduct(locals.read(span)))
        def assign(next: State)(using BlockBuilder, DslSourcePosition): Unit =
          locals.assign(Tuple.fromProductTyped(next))

    private[dsl] def snapshot(name: String, next: State)(using
        BlockBuilder, DslSourcePosition
    ): State =
      val snapshots = fields.snapshot(name, Tuple.fromProductTyped(next), 0)
      ExpressionStaging.expression(mirror.fromProduct(snapshots))

private[dsl] trait ProductFoldLocals[State <: Product]:
  def read(span: SourceSpan): State
  def assign(next: State)(using BlockBuilder, DslSourcePosition): Unit
