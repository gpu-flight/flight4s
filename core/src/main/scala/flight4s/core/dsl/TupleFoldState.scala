package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.CudaType

/** Evidence that each state component is a typed device expression. */
sealed trait TupleFoldState[State <: Tuple]:
  private[dsl] def declare(name: String, initial: State, offset: Int)(using
      BlockBuilder, DslSourcePosition
  ): TupleFoldLocals[State]

  private[dsl] def snapshot(name: String, next: State, offset: Int)(using
      BlockBuilder, DslSourcePosition
  ): State

object TupleFoldState:
  given empty: TupleFoldState[EmptyTuple] with
    private[dsl] def declare(name: String, initial: EmptyTuple, offset: Int)(using
        BlockBuilder, DslSourcePosition
    ): TupleFoldLocals[EmptyTuple] = new TupleFoldLocals[EmptyTuple]:
      def read(span: SourceSpan): EmptyTuple = EmptyTuple
      def assign(next: EmptyTuple)(using BlockBuilder, DslSourcePosition): Unit = ()

    private[dsl] def snapshot(name: String, next: EmptyTuple, offset: Int)(using
        BlockBuilder, DslSourcePosition
    ): EmptyTuple = EmptyTuple

  given nonEmpty[A, Tail <: Tuple](using
      headType: CudaType[A], tail: TupleFoldState[Tail]
  ): TupleFoldState[Expr[A] *: Tail] with
    private[dsl] def declare(name: String, initial: Expr[A] *: Tail, offset: Int)(using
        BlockBuilder, DslSourcePosition
    ): TupleFoldLocals[Expr[A] *: Tail] =
      val head = local(s"${name}_$offset", initial.head)
      val rest = tail.declare(name, initial.tail, offset + 1)
      new TupleFoldLocals[Expr[A] *: Tail]:
        def read(span: SourceSpan): Expr[A] *: Tail = Load(head, span) *: rest.read(span)
        def assign(next: Expr[A] *: Tail)(using BlockBuilder, DslSourcePosition): Unit =
          head := next.head
          rest.assign(next.tail)

    private[dsl] def snapshot(name: String, next: Expr[A] *: Tail, offset: Int)(using
        BlockBuilder, DslSourcePosition
    ): Expr[A] *: Tail =
      val head = let(s"${name}_next_$offset", next.head)
      val rest = tail.snapshot(name, next.tail, offset + 1)
      head *: rest

private[dsl] trait TupleFoldLocals[State <: Tuple]:
  def read(span: SourceSpan): State
  def assign(next: State)(using BlockBuilder, DslSourcePosition): Unit
