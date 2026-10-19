package flight4s.frontend

import scala.annotation.experimental
import flight4s.core.dsl.CudaDsl
import flight4s.core.ir.*

/** Opt-in source syntax; marker operations must never execute on the host. */
object ScalaKernel:
  export CudaDsl.{input, output, inOut, value, params}

  type DeviceBindingOf[Param] = Param match
    case ScalarParam[t] => t
    case BufferParam[t, mode] => DeviceArray[t, mode]

  type DeviceBindingsOf[Params <: Tuple] <: Tuple = Params match
    case EmptyTuple => EmptyTuple
    case head *: tail => DeviceBindingOf[head] *: DeviceBindingsOf[tail]

  final class DeviceArray[T, Mode <: AccessMode] private[frontend] ():
    def apply(index: Int): T = markerOnly()
    def update(index: Int, value: T)(using Mode =:= ReadWrite): Unit = markerOnly()

  /** Per-block storage marker, distinct from a global-memory kernel argument. */
  final class DeviceSharedArray[T] private[frontend] ():
    def apply(index: Int): T = markerOnly()
    def update(index: Int, value: T): Unit = markerOnly()

  /** Per-block rank-2 storage marker addressed by logical row and column. */
  final class DeviceSharedArray2D[T] private[frontend] ():
    def apply(row: Int, column: Int): T = markerOnly()
    def update(row: Int, column: Int, value: T): Unit = markerOnly()

  /** Per-block rank-3 storage marker addressed by logical layer, row and column. */
  final class DeviceSharedArray3D[T] private[frontend] ():
    def apply(layer: Int, row: Int, column: Int): T = markerOnly()
    def update(layer: Int, row: Int, column: Int, value: T): Unit = markerOnly()

  /** Declares uninitialized static storage with a positive compile-time element count. */
  def sharedArray[T](elementCount: Int): DeviceSharedArray[T] = markerOnly()

  /** Declares uninitialized static rank-2 storage; the physical row stride equals columns. */
  def sharedArray2D[T](rows: Int, columns: Int): DeviceSharedArray2D[T] = markerOnly()

  /** Declares uninitialized static rank-3 storage; the physical row stride equals columns. */
  def sharedArray3D[T](depth: Int, rows: Int, columns: Int): DeviceSharedArray3D[T] = markerOnly()

  /** Per-block sizeless storage marker; the element count is a launch-time decision. */
  final class DeviceDynamicSharedArray[T] private[frontend] ():
    def apply(index: Int): T = markerOnly()
    def update(index: Int, value: T): Unit = markerOnly()

  /** Declares the kernel's single dynamic shared array; the launch supplies dynamicSharedMemoryBytes. */
  def dynamicSharedArray[T](): DeviceDynamicSharedArray[T] = markerOnly()

  /** Block-wide execution and memory barrier; not a lock or a grid-wide barrier. */
  def barrier(): Unit = markerOnly()

  object sync:
    /** Block-wide execution and memory barrier; equivalent to barrier(). */
    def block(): Unit = markerOnly()

    /** Executes a scoped device body, then one block barrier; no entry barrier or lock. */
    def blockAfter(body: => Unit): Unit = markerOnly()

    /** Bit pattern naming all 32 lanes of the calling warp. */
    inline val fullMask = -1

    /** Warp barrier among exactly the lanes of the compile-time constant mask; not a lane filter. */
    def warp(mask: Int): Unit = markerOnly()

    /** Executes a scoped device body, then one warp barrier; no entry barrier or lane guard. */
    def warpAfter(mask: Int)(body: => Unit): Unit = markerOnly()

  /** Warp tree-reduction sum over aligned mask groups; a shuffle tree, not an ordered foldLeft sum. */
  def warpReduceSum[T](mask: Int, value: T): T = markerOnly()

  /** Same reduction over independent aligned lane groups of the compile-time width. */
  def warpReduceSum[T](mask: Int, value: T, width: Int): T = markerOnly()

  /** Warp tree reduction with an explicit combine; subtree order is preserved in every lane. */
  def warpReduceTree[T](mask: Int, value: T)(combine: (T, T) => T): T = markerOnly()

  /** Same tree over independent aligned lane groups of the compile-time width. */
  def warpReduceTree[T](mask: Int, value: T, width: Int)(combine: (T, T) => T): T = markerOnly()

  /** Receives the value the source lane passed into this call; a register exchange, not a barrier. */
  def warpShuffle[T](mask: Int, value: T, sourceLane: Int): T = markerOnly()
  def warpShuffle[T](mask: Int, value: T, sourceLane: Int, width: Int): T = markerOnly()

  /** Lane i receives from lane i - delta inside its width segment; low lanes keep their own value. */
  def warpShuffleUp[T](mask: Int, value: T, delta: Int): T = markerOnly()
  def warpShuffleUp[T](mask: Int, value: T, delta: Int, width: Int): T = markerOnly()

  /** Lane i receives from lane i + delta inside its width segment; high lanes keep their own value. */
  def warpShuffleDown[T](mask: Int, value: T, delta: Int): T = markerOnly()
  def warpShuffleDown[T](mask: Int, value: T, delta: Int, width: Int): T = markerOnly()

  /** Exchanges with lane i XOR laneMask inside the width segment; one butterfly step. */
  def warpShuffleXor[T](mask: Int, value: T, laneMask: Int): T = markerOnly()
  def warpShuffleXor[T](mask: Int, value: T, laneMask: Int, width: Int): T = markerOnly()

  /** One-shot atomic view over a mutable element; consumed immediately by one operation. */
  final class DeviceAtomic[T] private[frontend] ():
    /** Device-scope relaxed read-modify-write add; no result. */
    def add(value: T): Unit = markerOnly()

    /** The same transaction, binding CUDA's returned old value, not the new one. */
    def fetchAdd(value: T): T = markerOnly()

  /** Wraps a read-write buffer element or rank-1 shared element for one atomic operation. */
  def atomic[T](target: T): DeviceAtomic[T] = markerOnly()

  /** One-shot scoped atomic view; every operation names its memory order explicitly. */
  final class DeviceScopedAtomic[T] private[frontend] ():
    def store(value: T, order: MemoryOrder): Unit = markerOnly()
    def load(order: MemoryOrder): T = markerOnly()
    def exchange(value: T, order: MemoryOrder): T = markerOnly()
    def fetchAdd(value: T, order: MemoryOrder): T = markerOnly()
    def fetchSub(value: T, order: MemoryOrder): T = markerOnly()

  /** Device-scope view over a global read-write element; orders against every GPU thread. */
  def atomicDevice[T](target: T): DeviceScopedAtomic[T] = markerOnly()

  /** Block-scope view over a global or rank-1 shared element; orders within one block only. */
  def atomicBlock[T](target: T): DeviceScopedAtomic[T] = markerOnly()

  /** Per-block reduction scratch bound to one compile-time 1-D launch shape. */
  final class DeviceBlockReduction[T] private[frontend] ():
    def sum(value: T): T = markerOnly()
    def reduceTree(value: T)(combine: (T, T) => T): T = markerOnly()

  /** Declares power-of-two shared reduction scratch for exactly threads lanes per block. */
  def blockReduction[T](threads: Int): DeviceBlockReduction[T] = markerOnly()

  /** Lazy serial device traversal syntax, not a materialized Scala collection. */
  final class DeviceTraversal[T] private[frontend] ():
    def map[U](transform: T => U): DeviceTraversal[U] = markerOnly()
    def flatMap[U](transform: T => DeviceTraversal[U]): DeviceTraversal[U] = markerOnly()
    def withFilter(predicate: T => Boolean): DeviceTraversal[T] = markerOnly()
    def foreach(body: T => Unit): Unit = markerOnly()
    def foldLeft[A](initial: A)(step: (A, T) => A): A = markerOnly()

  /** Captures half-open unit-stride bounds at this device statement. */
  def deviceRange(from: Int, until: Int): DeviceTraversal[Int] = markerOnly()

  private def markerOnly[T](): T =
    throw IllegalStateException("ScalaKernel device markers can only be used inside ScalaKernel.kernel")

  object threadIdx:
    def x: Int = markerOnly()
    def y: Int = markerOnly()
    def z: Int = markerOnly()

  object blockIdx:
    def x: Int = markerOnly()
    def y: Int = markerOnly()
    def z: Int = markerOnly()

  object blockDim:
    def x: Int = markerOnly()
    def y: Int = markerOnly()
    def z: Int = markerOnly()

  object gridDim:
    def x: Int = markerOnly()
    def y: Int = markerOnly()
    def z: Int = markerOnly()

  @experimental
  inline def kernel[Args <: Tuple, Params <: Tuple](name: String,
      signature: KernelSignature[Args] { type Bindings = Params })(
      inline body: DeviceBindingsOf[Params] => Unit): Kernel[Args] =
    ${ ScalaKernelMacro.build[Args, Params]('name, 'signature, 'body) }
