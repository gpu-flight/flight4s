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
