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

  /** Declares uninitialized static storage with a positive compile-time element count. */
  def sharedArray[T](elementCount: Int): DeviceSharedArray[T] = markerOnly()

  /** Block-wide execution and memory barrier; not a lock or a grid-wide barrier. */
  def barrier(): Unit = markerOnly()

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
