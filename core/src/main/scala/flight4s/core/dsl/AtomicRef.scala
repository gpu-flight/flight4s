package flight4s.core.dsl

import flight4s.core.dsl.CudaDsl.BlockBuilder
import flight4s.core.ir.*
import flight4s.core.types.{AtomicIntegralType, AtomicValueType}

/** A staged reference, not a host resource. Each operation is one explicit device statement. */
final class AtomicRef[T, Space <: Global | Shared] private[dsl] (
    target: Place[T, Space, ReadWrite],
    scope: AtomicScope,
    atomicType: AtomicValueType[T]
):
  def load(name: String, order: MemoryOrder)(using BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.Load, Vector.empty, order)

  def store(value: Expr[T], order: MemoryOrder)(using builder: BlockBuilder, position: DslSourcePosition): Unit =
    builder.append(AtomicStore(target, value, atomicType, order, scope, position.span))

  def exchange(name: String, value: Expr[T], order: MemoryOrder)(using BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.Exchange, Vector(value), order)

  def fetchAdd(name: String, value: Expr[T], order: MemoryOrder)(using BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.FetchAdd, Vector(value), order)

  def fetchSub(name: String, value: Expr[T], order: MemoryOrder)(using BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.FetchSub, Vector(value), order)

  def fetchMin(name: String, value: Expr[T], order: MemoryOrder)(using AtomicIntegralType[T], BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.FetchMin, Vector(value), order)

  def fetchMax(name: String, value: Expr[T], order: MemoryOrder)(using AtomicIntegralType[T], BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.FetchMax, Vector(value), order)

  def fetchAnd(name: String, value: Expr[T], order: MemoryOrder)(using AtomicIntegralType[T], BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.FetchAnd, Vector(value), order)

  def fetchOr(name: String, value: Expr[T], order: MemoryOrder)(using AtomicIntegralType[T], BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.FetchOr, Vector(value), order)

  def fetchXor(name: String, value: Expr[T], order: MemoryOrder)(using AtomicIntegralType[T], BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.FetchXor, Vector(value), order)

  /** Returns the observed old value. For integral types, equality with expected means success. */
  def compareExchange(name: String, expected: Expr[T], desired: Expr[T], successOrder: MemoryOrder,
      failureOrder: MemoryOrder)(using AtomicIntegralType[T], BlockBuilder, DslSourcePosition): Expr[T] =
    result(name, AtomicOperation.CompareExchange, Vector(expected, desired), successOrder, Some(failureOrder))

  private def result(name: String, operation: AtomicOperation, operands: Vector[Expr[T]], order: MemoryOrder,
      failureOrder: Option[MemoryOrder] = None)(using builder: BlockBuilder, position: DslSourcePosition): Expr[T] =
    val local = LocalVariable(name, atomicType, position.span)
    builder.append(AtomicResult(local, target, operation, operands, atomicType, order, scope, failureOrder, position.span))
    Load(local, position.span)
