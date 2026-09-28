package flight4s.core.ir

import flight4s.core.abi.*
import flight4s.core.launch.{Block as LaunchBlock, LaunchConfig}

trait DeviceBuffer[T]:
  private[flight4s] def deviceAddress: DeviceAddress

type KernelArgumentOf[Param] = Param match
  case ScalarParam[value] => value
  case BufferParam[value, mode] => DeviceBuffer[value]

type KernelArgumentsOf[Params <: Tuple] <: Tuple = Params match
  case EmptyTuple => EmptyTuple
  case head *: tail => KernelArgumentOf[head] *: KernelArgumentsOf[tail]

sealed trait KernelParamTuple[Params <: Tuple]:
  def toVector(params: Params): Vector[KernelParam]
  def abiDescriptors(params: Params): Vector[KernelArgumentAbi]
  private[flight4s] def pack(
      params: Params,
      arguments: KernelArgumentsOf[Params]
  ): Vector[PackedKernelArgument]

sealed trait KernelParamAbi[Param <: KernelParam]:
  def descriptor(param: Param): KernelArgumentAbi
  private[flight4s] def pack(
      param: Param,
      argument: KernelArgumentOf[Param]
  ): PackedKernelArgument

object KernelParamAbi:
  given scalarParamAbi[T]: KernelParamAbi[ScalarParam[T]] with
    override def descriptor(param: ScalarParam[T]): KernelArgumentAbi =
      KernelArgumentAbi(param.name, param.scalarAbi.abiType)

    override private[flight4s] def pack(
        param: ScalarParam[T],
        argument: T
    ): PackedKernelArgument =
      PackedKernelArgument(descriptor(param), param.scalarAbi.encode(argument))

  given bufferParamAbi[
      T,
      Mode <: AccessMode
  ]: KernelParamAbi[BufferParam[T, Mode]] with
    override def descriptor(
        param: BufferParam[T, Mode]
    ): KernelArgumentAbi =
      KernelArgumentAbi(param.name, CudaAbiType.DevicePointer64)

    override private[flight4s] def pack(
        param: BufferParam[T, Mode],
        argument: DeviceBuffer[T]
    ): PackedKernelArgument =
      PackedKernelArgument(
        descriptor(param),
        DeviceAddress.encode(argument.deviceAddress)
      )

object KernelParamTuple:
  given emptyTuple: KernelParamTuple[EmptyTuple] with
    override def toVector(params: EmptyTuple): Vector[KernelParam] =
      Vector.empty
    override def abiDescriptors(
        params: EmptyTuple
    ): Vector[KernelArgumentAbi] =
      Vector.empty
    override private[flight4s] def pack(
        params: EmptyTuple,
        arguments: EmptyTuple
    ): Vector[PackedKernelArgument] =
      Vector.empty

  given nonEmptyTuple[
      Head <: KernelParam,
      Tail <: Tuple
  ](using
      headAbi: KernelParamAbi[Head],
      tailParams: KernelParamTuple[Tail]
  ): KernelParamTuple[Head *: Tail] with
    override def toVector(params: Head *: Tail): Vector[KernelParam] =
      params.head +: tailParams.toVector(params.tail)
    override def abiDescriptors(
        params: Head *: Tail
    ): Vector[KernelArgumentAbi] =
      headAbi.descriptor(params.head) +:
        tailParams.abiDescriptors(params.tail)
    override private[flight4s] def pack(
        params: Head *: Tail,
        arguments: KernelArgumentsOf[Head *: Tail]
    ): Vector[PackedKernelArgument] =
      headAbi.pack(params.head, arguments.head) +:
        tailParams.pack(params.tail, arguments.tail)

sealed trait KernelSignature[Args <: Tuple]:
  type Bindings <: Tuple

  def bindings: Bindings
  def parameters: Vector[KernelParam]
  def abiDescriptors: Vector[KernelArgumentAbi]
  private[flight4s] def pack(arguments: Args): PackedKernelArguments

object KernelSignature:
  private final class Impl[Params <: Tuple](
      override val bindings: Params,
      tuple: KernelParamTuple[Params]
  ) extends KernelSignature[KernelArgumentsOf[Params]]:
    override type Bindings = Params
    override val parameters: Vector[KernelParam] = tuple.toVector(bindings)
    override val abiDescriptors: Vector[KernelArgumentAbi] =
      tuple.abiDescriptors(bindings)
    override private[flight4s] def pack(
        arguments: KernelArgumentsOf[Params]
    ): PackedKernelArguments =
      PackedKernelArguments(tuple.pack(bindings, arguments))

  def fromTuple[Params <: Tuple](
      bindings: Params
  )(using tuple: KernelParamTuple[Params])
      : KernelSignature[KernelArgumentsOf[Params]] { type Bindings = Params } =
    new Impl(bindings, tuple)

final case class KernelIR[Args <: Tuple](
    name: String,
    signature: KernelSignature[Args],
    body: Block,
    sharedMemory: Vector[SharedArray[?, ?]] = Vector.empty,
    requiredBlock: Option[LaunchBlock] = None
):
  def params: Vector[KernelParam] = signature.parameters

final case class Kernel[Args <: Tuple](ir: KernelIR[Args]):
  def name: String = ir.name
  def signature: KernelSignature[Args] = ir.signature
  def params: Vector[KernelParam] = ir.params
  def body: Block = ir.body
  def sharedMemory: Vector[SharedArray[?, ?]] = ir.sharedMemory
  def requiredBlock: Option[LaunchBlock] = ir.requiredBlock

  def requiringBlock(block: LaunchBlock): Kernel[Args] =
    copy(ir = ir.copy(requiredBlock = Some(block)))

  def bind(arguments: Args): KernelInvocation[Args] =
    KernelInvocation(this, arguments)

final case class KernelInvocation[Args <: Tuple](
    kernel: Kernel[Args],
    arguments: Args
):
  private[flight4s] def packedArguments: PackedKernelArguments =
    kernel.signature.pack(arguments)

  private[flight4s] def nativeArguments: NativeArgumentStorage =
    NativeArgumentStorage.materialize(packedArguments)

  private[flight4s] def nativeLaunchRequest(
      config: LaunchConfig
  ): NativeLaunchRequest =
    NativeLaunchRequest.materialize(config, nativeArguments)
