package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.CudaType

private[dsl] object PairFold:
  def stage[A, B, T](name: String, initial: (Expr[A], Expr[B]))(
      step: ((Expr[A], Expr[B]), Expr[T]) => (Expr[A], Expr[B])
  )(iterate: (Expr[T] => (BlockBuilder ?=> Unit)) => Unit)(using
      CudaType[A], CudaType[B], BlockBuilder, DslSourcePosition
  ): (Expr[A], Expr[B]) =
    val first = local(s"${name}_0", initial._1)
    val second = local(s"${name}_1", initial._2)
    iterate { value =>
      val next = ExpressionStaging.expression(step((first.read, second.read), value))
      // Both expressions may read either old component. Snapshot before either store.
      val nextFirst = let(s"${name}_next_0", next._1)
      val nextSecond = let(s"${name}_next_1", next._2)
      first := nextFirst
      second := nextSecond
    }
    val span = summon[DslSourcePosition].span
    (Load(first, span), Load(second, span))
