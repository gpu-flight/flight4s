package flight4s.runtime.cuda

import java.nio.{ByteBuffer, ByteOrder}
import java.security.MessageDigest
import java.util.HexFormat
import scala.util.control.NonFatal

import flight4s.core.abi.{DeviceAddress, NativeArgumentStorage, NativeLaunchRequest}
import flight4s.core.codegen.GeneratedKernel
import flight4s.core.compiler.{
  ComputeCapability,
  NvrtcArtifact,
  NvrtcCompilationInput,
  NvrtcKernelInput,
  NvrtcSourceProvenance
}
import flight4s.core.ir.{DeviceBuffer, KernelInvocation, KernelSignature}
import flight4s.core.launch.{Block as LaunchBlock, LaunchConfig}
import flight4s.core.types.CudaType
import flight4s.core.unsafe.raw.{RawCudaInvocation, RawCudaKernel}
import flight4s.runtime.cuda.internal.*

final case class CudaDriverFailure(
    operation: String,
    resultCode: Int,
    resultName: String,
    resultDescription: String,
    infoLog: String,
    errorLog: String
):
  def message: String =
    s"$operation failed with $resultName ($resultCode): $resultDescription"

final class CudaDriverException(
    val failure: CudaDriverFailure
) extends RuntimeException(failure.message)

final case class CudaFunctionAttributes(
    maxThreadsPerBlock: Int,
    staticSharedMemoryBytes: Int,
    constantMemoryBytes: Int,
    localMemoryBytes: Int,
    registersPerThread: Int
):
  require(
    maxThreadsPerBlock > 0,
    s"CUDA function maximum threads per block must be positive: $maxThreadsPerBlock"
  )
  require(
    staticSharedMemoryBytes >= 0,
    s"CUDA function static shared memory must not be negative: $staticSharedMemoryBytes"
  )
  require(
    constantMemoryBytes >= 0,
    s"CUDA function constant memory must not be negative: $constantMemoryBytes"
  )
  require(
    localMemoryBytes >= 0,
    s"CUDA function local memory must not be negative: $localMemoryBytes"
  )
  require(
    registersPerThread >= 0,
    s"CUDA function registers per thread must not be negative: $registersPerThread"
  )

sealed trait CudaLaunchFailure:
  def message: String

object CudaLaunchFailure:
  final case class BlockShapeMismatch(
      kernelName: String,
      required: LaunchBlock,
      configured: LaunchBlock
  ) extends CudaLaunchFailure:
    override val message: String =
      s"kernel $kernelName requires block (${required.x}, ${required.y}, ${required.z}), " +
        s"but the configured block is (${configured.x}, ${configured.y}, ${configured.z})"

  final case class InvocationMismatch(
      expectedKernel: String,
      actualKernel: String
  ) extends CudaLaunchFailure:
    override val message: String =
      s"kernel invocation $actualKernel does not belong to $expectedKernel"

  final case class DynamicSharedMemoryRequired(
      kernelName: String,
      elementSizeBytes: Int
  ) extends CudaLaunchFailure:
    override val message: String =
      s"kernel $kernelName requires dynamic shared memory for " +
        s"$elementSizeBytes-byte elements"

  final case class InvalidDynamicSharedMemorySize(
      kernelName: String,
      configuredBytes: Int,
      elementSizeBytes: Int,
      elementAlignmentBytes: Int
  ) extends CudaLaunchFailure:
    override val message: String =
      s"kernel $kernelName dynamic shared memory size " +
        s"$configuredBytes is incompatible with $elementSizeBytes-byte " +
        s"elements aligned to $elementAlignmentBytes bytes"

  final case class StreamContextMismatch(
      kernelName: String
  ) extends CudaLaunchFailure:
    override val message: String =
      s"CUDA stream does not belong to the context for kernel $kernelName"

  final case class BufferContextMismatch(
      kernelName: String,
      argumentIndex: Int
  ) extends CudaLaunchFailure:
    override val message: String =
      s"CUDA device buffer at argument $argumentIndex does not belong to " +
        s"the context for kernel $kernelName"

  final case class Driver(
      failure: CudaDriverFailure
  ) extends CudaLaunchFailure:
    override def message: String = failure.message

enum CudaStreamMode(private[cuda] val nativeFlags: Int):
  case Default extends CudaStreamMode(0)
  case NonBlocking extends CudaStreamMode(1)

enum CudaEventMode(private[cuda] val nativeFlags: Int):
  case Completion extends CudaEventMode(2)
  case BlockingCompletion extends CudaEventMode(3)

private[cuda] trait CudaInFlightResource:
  private[cuda] def acquireInFlight(): InFlightLease[CudaDriverFailure]

private final case class PtxModuleKey private (hex: String)

private object PtxModuleKey:
  def derive(ptx: IArray[Byte]): PtxModuleKey =
    val bytes = IArray.genericWrapArray(ptx).toArray
    PtxModuleKey(
      HexFormat.of().formatHex(
        MessageDigest.getInstance("SHA-256").digest(bytes)
      )
    )

final class CudaContext private (
    val deviceOrdinal: Int,
    val computeCapability: ComputeCapability,
    private val handle: Long,
    private val backend: CudaDriverBackend
) extends AutoCloseable:
  private val lifecycleLock = new Object
  private val defaultStreamTracker = InFlightTracker[CudaDriverFailure]()
  private var closed = false
  private var resources = Vector.empty[AutoCloseable]
  private var nativeModules = Map.empty[PtxModuleKey, SharedCudaModule]

  def isOpen: Boolean =
    lifecycleLock.synchronized(!closed)

  def load(
      artifact: NvrtcArtifact
  ): Either[CudaDriverFailure, CudaModule] =
    lifecycleLock.synchronized {
      requireOpen()
      val key = PtxModuleKey.derive(artifact.ptx)
      nativeModules.get(key) match
        case Some(shared) => Right(newModule(artifact, shared))
        case None =>
          val result = backend.loadPtx(handle, artifact.ptx)
          if result.status.succeeded then
            require(
              result.handle != 0L,
              "successful CUDA module load returned a null handle"
            )
            val shared = new SharedCudaModule(
              context = this,
              key = key,
              handle = result.handle,
              backend = backend
            )
            nativeModules += key -> shared
            Right(newModule(artifact, shared))
          else
            Left(
              CudaDriverFailure.fromStatus(
                "CUDA PTX module load",
                result.status
              )
            )
    }

  private def newModule(
      artifact: NvrtcArtifact,
      shared: SharedCudaModule
  ): CudaModule =
    shared.retain()
    val module = new CudaModule(this, artifact, shared, backend)
    resources :+= module
    module

  private[cuda] def removeNativeModule(
      key: PtxModuleKey,
      shared: SharedCudaModule
  ): Unit =
    nativeModules.get(key).foreach { current =>
      if current eq shared then nativeModules -= key
    }

  def allocate[T](
      elementCount: Int
  )(using valueType: CudaType[T])
      : Either[CudaDriverFailure, CudaDeviceBuffer[T]] =
    lifecycleLock.synchronized {
      requireOpen()
      require(
        elementCount > 0,
        s"CUDA device buffer element count must be positive: $elementCount"
      )
      val sizeBytes =
        Math.multiplyExact(
          elementCount.toLong,
          valueType.sizeBytes.toLong
        )
      val result = backend.allocateDeviceMemory(handle, sizeBytes)
      if result.status.succeeded then
        require(
          result.handle != 0L,
          "successful CUDA device allocation returned a null address"
        )
        val buffer = new CudaDeviceBuffer(
          context = this,
          valueType = valueType,
          elementCount = elementCount,
          sizeBytes = sizeBytes,
          handle = result.handle,
          backend = backend
        )
        resources :+= buffer
        Right(buffer)
      else
        Left(
          CudaDriverFailure.fromStatus(
            s"CUDA device allocation of $sizeBytes bytes",
            result.status
          )
        )
    }

  def allocatePinned[T](
      elementCount: Int
  )(using valueType: CudaType[T])
      : Either[CudaDriverFailure, CudaPinnedBuffer[T]] =
    lifecycleLock.synchronized {
      requireOpen()
      require(
        elementCount > 0,
        s"CUDA pinned buffer element count must be positive: $elementCount"
      )
      val sizeBytes = Math.multiplyExact(
        elementCount.toLong,
        valueType.sizeBytes.toLong
      )
      require(
        sizeBytes <= Int.MaxValue,
        s"CUDA pinned buffers require at most ${Int.MaxValue} bytes: " +
          sizeBytes
      )
      val result = backend.allocatePinnedMemory(handle, sizeBytes)
      if result.status.succeeded then
        require(
          result.handle != 0L,
          "successful CUDA pinned allocation returned a null address"
        )
        require(
          result.storage != null && result.storage.isDirect,
          "successful CUDA pinned allocation returned invalid storage"
        )
        require(
          result.storage.position() == 0 &&
            result.storage.limit() == sizeBytes.toInt &&
            result.storage.capacity() == sizeBytes.toInt,
          "successful CUDA pinned allocation returned an inexact view"
        )
        val buffer = new CudaPinnedBuffer(
          context = this,
          valueType = valueType,
          elementCount = elementCount,
          sizeBytes = sizeBytes,
          handle = result.handle,
          storage = result.storage.order(ByteOrder.nativeOrder()),
          backend = backend
        )
        resources :+= buffer
        Right(buffer)
      else
        Left(
          CudaDriverFailure.fromStatus(
            s"CUDA pinned allocation of $sizeBytes bytes",
            result.status
          )
        )
    }

  def createStream(
      mode: CudaStreamMode = CudaStreamMode.NonBlocking
  ): Either[CudaDriverFailure, CudaStream] =
    lifecycleLock.synchronized {
      requireOpen()
      val result = backend.createStream(handle, mode.nativeFlags)
      if result.status.succeeded then
        require(
          result.handle != 0L,
          "successful CUDA stream creation returned a null handle"
        )
        val stream = new CudaStream(
          context = this,
          mode = mode,
          handle = result.handle,
          backend = backend
        )
        resources :+= stream
        Right(stream)
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA stream creation",
            result.status
          )
        )
    }

  def createEvent(
      mode: CudaEventMode = CudaEventMode.Completion
  ): Either[CudaDriverFailure, CudaEvent] =
    lifecycleLock.synchronized {
      requireOpen()
      val result = backend.createEvent(handle, mode.nativeFlags)
      if result.status.succeeded then
        require(
          result.handle != 0L,
          "successful CUDA event creation returned a null handle"
        )
        val event = new CudaEvent(
          context = this,
          mode = mode,
          handle = result.handle,
          backend = backend
        )
        resources :+= event
        Right(event)
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA event creation",
            result.status
          )
        )
    }

  def synchronize(): Either[CudaDriverFailure, Unit] =
    lifecycleLock.synchronized {
      requireOpen()
      val status = backend.synchronizeContext(handle)
      if status.succeeded then completeTrackedWork()
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA context synchronization",
            status
          )
        )
    }

  override def close(): Unit =
    lifecycleLock.synchronized {
      if !closed then
        if defaultStreamTracker.hasPending then
          val synchronization = backend.synchronizeContext(handle)
          if !synchronization.succeeded then
            throw CudaDriverException(
              CudaDriverFailure.fromStatus(
                "CUDA context synchronization during close",
                synchronization
              )
            )
          completeTrackedWork() match
            case Left(failure) => throw CudaDriverException(failure)
            case Right(()) => ()
        resources.reverse.foreach(_.close())
        require(
          nativeModules.isEmpty,
          "CUDA module cache retained a native module after wrapper cleanup"
        )
        val status = backend.releasePrimaryContext(deviceOrdinal)
        if !status.succeeded then
          throw CudaDriverException(
            CudaDriverFailure.fromStatus(
              "CUDA primary context release",
              status
            )
          )
        closed = true
    }

  private[cuda] def synchronizedLifecycle[A](body: => A): A =
    lifecycleLock.synchronized(body)

  private[cuda] def requireOpen(): Unit =
    if closed then
      throw IllegalStateException("CUDA context is closed")

  private[cuda] def unregister(resource: AutoCloseable): Unit =
    resources = resources.filterNot(_ eq resource)

  private[cuda] def submitDefaultTracked(
      resources: Vector[CudaInFlightResource]
  )(
      submit: => NativeCudaDriverStatus
  ): NativeCudaDriverStatus =
    lifecycleLock.synchronized {
      requireOpen()
      val batch = acquireBatch(resources)
      try
        val status = submit
        if status.succeeded then defaultStreamTracker.add(batch)
        else releaseFailedSubmission(batch)
        status
      catch
        case NonFatal(error) =>
          releaseFailedSubmission(batch)
          throw error
    }

  private def completeTrackedWork()
      : Either[CudaDriverFailure, Unit] =
    var firstFailure = Option.empty[CudaDriverFailure]

    def retainFirst(result: Either[CudaDriverFailure, Unit]): Unit =
      result match
        case Left(failure) if firstFailure.isEmpty =>
          firstFailure = Some(failure)
        case _ => ()

    retainFirst(defaultStreamTracker.completePending())
    resources.foreach {
      case stream: CudaStream =>
        retainFirst(stream.completeAfterContextSynchronization())
      case _ => ()
    }
    firstFailure.toLeft(())

  private def acquireBatch(
      resources: Vector[CudaInFlightResource]
  ): InFlightBatch[CudaDriverFailure] =
    val uniqueResources = resources.foldLeft(
      Vector.empty[CudaInFlightResource]
    ) { (unique, resource) =>
      if unique.exists(_ eq resource) then unique
      else unique :+ resource
    }
    InFlightBatch(uniqueResources.map(_.acquireInFlight()))

  private def releaseFailedSubmission(
      batch: InFlightBatch[CudaDriverFailure]
  ): Unit =
    batch.complete() match
      case Left(failure) => throw CudaDriverException(failure)
      case Right(()) => ()

  private[flight4s] def nativeHandle: Long =
    lifecycleLock.synchronized {
      requireOpen()
      handle
    }

object CudaContext:
  def open(
      deviceOrdinal: Int
  ): Either[CudaDriverFailure, CudaContext] =
    open(deviceOrdinal, NativeCudaDriver)

  private[cuda] def open(
      deviceOrdinal: Int,
      backend: CudaDriverBackend
  ): Either[CudaDriverFailure, CudaContext] =
    require(
      deviceOrdinal >= 0,
      "CUDA device ordinal must not be negative"
    )
    val result = backend.retainPrimaryContext(deviceOrdinal)
    if result.status.succeeded then
      require(
        result.handle != 0L,
        "successful CUDA context retain returned a null handle"
      )
      Right(
        new CudaContext(
          deviceOrdinal = result.deviceOrdinal,
          computeCapability = ComputeCapability(
            result.computeCapabilityMajor,
            result.computeCapabilityMinor
          ),
          handle = result.handle,
          backend = backend
        )
      )
    else
      Left(
        CudaDriverFailure.fromStatus(
          "CUDA primary context retain",
          result.status
        )
      )

final class CudaStream private[cuda] (
    val context: CudaContext,
    val mode: CudaStreamMode,
    private val handle: Long,
    private val backend: CudaDriverBackend
) extends AutoCloseable:
  private var closed = false
  private val tracker = InFlightTracker[CudaDriverFailure]()

  def isOpen: Boolean =
    context.synchronizedLifecycle(!closed && context.isOpen)

  def synchronize(): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      requireOpen()
      val status = backend.synchronizeStream(
        context.nativeHandle,
        handle
      )
      if status.succeeded then tracker.completePending()
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA stream synchronization",
            status
          )
        )
    }

  def waitFor(event: CudaEvent): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      requireOpen()
      require(
        event.context eq context,
        "CUDA event and stream must belong to the same context"
      )
      event.requireOpen()
      val status = backend.waitForEvent(
        context.nativeHandle,
        handle,
        event.nativeHandle
      )
      if status.succeeded then Right(())
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA stream event wait",
            status
          )
        )
    }

  override def close(): Unit =
    context.synchronizedLifecycle {
      if !closed then
        context.requireOpen()
        if tracker.hasPending then
          val synchronization = backend.synchronizeStream(
            context.nativeHandle,
            handle
          )
          if !synchronization.succeeded then
            throw CudaDriverException(
              CudaDriverFailure.fromStatus(
                "CUDA stream synchronization during close",
                synchronization
              )
            )
          tracker.completePending() match
            case Left(failure) => throw CudaDriverException(failure)
            case Right(()) => ()
        val status = backend.destroyStream(
          context.nativeHandle,
          handle
        )
        if !status.succeeded then
          throw CudaDriverException(
            CudaDriverFailure.fromStatus(
              "CUDA stream destruction",
              status
            )
          )
        closed = true
        context.unregister(this)
    }

  private[cuda] def requireOpen(): Unit =
    context.requireOpen()
    if closed then
      throw IllegalStateException("CUDA stream is closed")

  private[cuda] def nativeHandle: Long =
    context.synchronizedLifecycle {
      requireOpen()
      handle
    }

  private[cuda] def submitTracked(
      resources: Vector[CudaInFlightResource]
  )(
      submit: Long => NativeCudaDriverStatus
  ): NativeCudaDriverStatus =
    context.synchronizedLifecycle {
      requireOpen()
      val uniqueResources = resources.foldLeft(
        Vector.empty[CudaInFlightResource]
      ) { (unique, resource) =>
        if unique.exists(_ eq resource) then unique
        else unique :+ resource
      }
      val batch = InFlightBatch(
        uniqueResources.map(_.acquireInFlight())
      )
      try
        val status = submit(handle)
        if status.succeeded then tracker.add(batch)
        else releaseFailedSubmission(batch)
        status
      catch
        case NonFatal(error) =>
          releaseFailedSubmission(batch)
          throw error
    }

  private[cuda] def completionSnapshot()
      : Vector[InFlightBatch[CudaDriverFailure]] =
    context.synchronizedLifecycle {
      requireOpen()
      tracker.snapshot()
    }

  private[cuda] def completeBatches(
      batches: Vector[InFlightBatch[CudaDriverFailure]]
  ): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      tracker.complete(batches)
    }

  private[cuda] def completeAfterContextSynchronization()
      : Either[CudaDriverFailure, Unit] =
    tracker.completePending()

  private def releaseFailedSubmission(
      batch: InFlightBatch[CudaDriverFailure]
  ): Unit =
    batch.complete() match
      case Left(failure) => throw CudaDriverException(failure)
      case Right(()) => ()

final class CudaEvent private[cuda] (
    val context: CudaContext,
    val mode: CudaEventMode,
    private val handle: Long,
    private val backend: CudaDriverBackend
) extends AutoCloseable:
  private var closed = false
  private var recordedCompletion = Option.empty[
    (CudaStream, Vector[InFlightBatch[CudaDriverFailure]])
  ]

  def isOpen: Boolean =
    context.synchronizedLifecycle(!closed && context.isOpen)

  def record(stream: CudaStream): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      requireOpen()
      require(
        stream.context eq context,
        "CUDA event and stream must belong to the same context"
      )
      stream.requireOpen()
      val status = backend.recordEvent(
        context.nativeHandle,
        handle,
        stream.nativeHandle
      )
      if status.succeeded then
        recordedCompletion = Some(
          stream -> stream.completionSnapshot()
        )
        Right(())
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA event recording",
            status
          )
        )
    }

  def query(): Either[CudaDriverFailure, Boolean] =
    context.synchronizedLifecycle {
      requireOpen()
      val result = backend.queryEvent(
        context.nativeHandle,
        handle
      )
      if result.status.succeeded && result.complete then
        completeRecorded().map(_ => true)
      else if result.status.succeeded then Right(false)
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA event query",
            result.status
          )
        )
    }

  def synchronize(): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      requireOpen()
      val status = backend.synchronizeEvent(
        context.nativeHandle,
        handle
      )
      if status.succeeded then completeRecorded()
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA event synchronization",
            status
          )
        )
    }

  override def close(): Unit =
    context.synchronizedLifecycle {
      if !closed then
        context.requireOpen()
        val status = backend.destroyEvent(
          context.nativeHandle,
          handle
        )
        if !status.succeeded then
          throw CudaDriverException(
            CudaDriverFailure.fromStatus(
              "CUDA event destruction",
              status
            )
          )
        recordedCompletion = None
        closed = true
        context.unregister(this)
    }

  private[cuda] def requireOpen(): Unit =
    context.requireOpen()
    if closed then
      throw IllegalStateException("CUDA event is closed")

  private[cuda] def nativeHandle: Long =
    context.synchronizedLifecycle {
      requireOpen()
      handle
    }

  private def completeRecorded(): Either[CudaDriverFailure, Unit] =
    val completion = recordedCompletion
    recordedCompletion = None
    completion match
      case Some((stream, batches)) => stream.completeBatches(batches)
      case None => Right(())

final class CudaPinnedBuffer[T] private[cuda] (
    val context: CudaContext,
    val valueType: CudaType[T],
    val elementCount: Int,
    val sizeBytes: Long,
    private val handle: Long,
    private val storage: ByteBuffer,
    private val backend: CudaDriverBackend
) extends AutoCloseable,
      CudaInFlightResource:
  private val lifetime = InFlightResourceState[CudaDriverFailure] { () =>
    val status = backend.freePinnedMemory(
      context.nativeHandle,
      handle
    )
    if status.succeeded then
      context.unregister(this)
      Right(())
    else
      Left(
        CudaDriverFailure.fromStatus(
          "CUDA pinned memory free",
          status
        )
      )
  }

  def isOpen: Boolean =
    context.synchronizedLifecycle(lifetime.isOpen && context.isOpen)

  def copyFrom(
      values: Array[T]
  )(using codec: CudaHostCodec[T]): Unit =
    requireHostCodec(codec)
    require(
      values.length == elementCount,
      s"host element count ${values.length} does not match " +
        s"pinned buffer element count $elementCount"
    )
    context.synchronizedLifecycle {
      requireOpen()
      requireIdle()
      codec.encodeInto(values, storage)
    }

  def toArray(using codec: CudaHostCodec[T]): Array[T] =
    requireHostCodec(codec)
    context.synchronizedLifecycle {
      requireOpen()
      requireIdle()
      codec.decode(storage, elementCount)
    }

  override def close(): Unit =
    context.synchronizedLifecycle {
      if !lifetime.isReleased then
        context.requireOpen()
        lifetime.requestClose() match
          case Left(failure) => throw CudaDriverException(failure)
          case Right(()) => ()
    }

  private[cuda] def requireOpen(): Unit =
    context.requireOpen()
    if !lifetime.isOpen then
      throw IllegalStateException("CUDA pinned buffer is closed")

  private[cuda] def requireIdle(): Unit =
    lifetime.requireIdle("CUDA pinned buffer")

  override private[cuda] def acquireInFlight()
      : InFlightLease[CudaDriverFailure] =
    requireOpen()
    lifetime.acquire()

  private[cuda] def transferView: ByteBuffer =
    transferView(0L, sizeBytes)

  private[cuda] def transferView(
      offsetBytes: Long,
      lengthBytes: Long
  ): ByteBuffer =
    context.synchronizedLifecycle {
      requireOpen()
      require(offsetBytes >= 0L, "pinned transfer offset must not be negative")
      require(lengthBytes > 0L, "pinned transfer size must be positive")
      val endBytes = Math.addExact(offsetBytes, lengthBytes)
      require(
        endBytes <= sizeBytes,
        s"pinned transfer range [$offsetBytes, $endBytes) exceeds " +
          s"buffer size $sizeBytes"
      )
      val view = storage.duplicate().order(ByteOrder.nativeOrder())
      view.position(Math.toIntExact(offsetBytes))
      view.limit(Math.toIntExact(endBytes))
      view.slice().order(ByteOrder.nativeOrder())
    }

  private def requireHostCodec(codec: CudaHostCodec[T]): Unit =
    require(
      codec.cudaType eq valueType,
      s"host codec ${codec.cudaType.cudaName} does not match " +
        s"pinned buffer type ${valueType.cudaName}"
    )

final class CudaDeviceBuffer[T] private[cuda] (
    val context: CudaContext,
    val valueType: CudaType[T],
    val elementCount: Int,
    val sizeBytes: Long,
    private val handle: Long,
    private val backend: CudaDriverBackend
) extends DeviceBuffer[T],
      AutoCloseable,
      CudaInFlightResource:
  private val lifetime = InFlightResourceState[CudaDriverFailure] { () =>
    val status = backend.freeDeviceMemory(
      context.nativeHandle,
      handle
    )
    if status.succeeded then
      context.unregister(this)
      Right(())
    else
      Left(
        CudaDriverFailure.fromStatus(
          "CUDA device memory free",
          status
        )
      )
  }

  def isOpen: Boolean =
    context.synchronizedLifecycle(lifetime.isOpen && context.isOpen)

  def copyFrom(
      values: Array[T]
  )(using codec: CudaHostCodec[T]): Either[CudaDriverFailure, Unit] =
    requireHostCodec(codec)
    require(
      values.length == elementCount,
      s"host element count ${values.length} does not match " +
        s"device buffer element count $elementCount"
    )
    requireWholeBufferHostCopy()
    val source = codec.encode(values)
    context.synchronizedLifecycle {
      requireOpen()
      requireIdle()
      copyResult(
        "CUDA host-to-device copy",
        backend.copyHostToDevice(
          context.nativeHandle,
          handle,
          0L,
          source
        )
      )
    }

  def copyFrom(
      source: CudaPinnedBuffer[T]
  ): Either[CudaDriverFailure, Unit] =
    requirePinnedBuffer(source)
    requireMatchingPinnedBufferSize(source)
    copyFrom(
      source = source,
      sourceOffset = 0,
      destinationOffset = 0,
      elementCount = elementCount
    )

  def copyFrom(
      source: CudaPinnedBuffer[T],
      sourceOffset: Int,
      destinationOffset: Int,
      elementCount: Int
  ): Either[CudaDriverFailure, Unit] =
    requirePinnedBuffer(source)
    requireElementRange(
      sourceOffset,
      elementCount,
      source.elementCount,
      "pinned source"
    )
    requireElementRange(
      destinationOffset,
      elementCount,
      this.elementCount,
      "device destination"
    )
    val sourceOffsetBytes = byteExtent(sourceOffset)
    val destinationOffsetBytes = byteExtent(destinationOffset)
    val copySizeBytes = byteExtent(elementCount)
    context.synchronizedLifecycle {
      requireOpen()
      source.requireOpen()
      requireIdle()
      source.requireIdle()
      copyResult(
        "CUDA pinned host-to-device copy",
        backend.copyHostToDevice(
          context.nativeHandle,
          handle,
          destinationOffsetBytes,
          source.transferView(sourceOffsetBytes, copySizeBytes)
        )
      )
    }

  def copyFromAsync(
      source: CudaPinnedBuffer[T],
      stream: CudaStream
  ): Either[CudaDriverFailure, Unit] =
    requirePinnedBuffer(source)
    requireMatchingPinnedBufferSize(source)
    copyFromAsync(
      source = source,
      sourceOffset = 0,
      destinationOffset = 0,
      elementCount = elementCount,
      stream = stream
    )

  def copyFromAsync(
      source: CudaPinnedBuffer[T],
      sourceOffset: Int,
      destinationOffset: Int,
      elementCount: Int,
      stream: CudaStream
  ): Either[CudaDriverFailure, Unit] =
    requirePinnedBuffer(source)
    requireStream(stream)
    requireElementRange(
      sourceOffset,
      elementCount,
      source.elementCount,
      "pinned source"
    )
    requireElementRange(
      destinationOffset,
      elementCount,
      this.elementCount,
      "device destination"
    )
    val sourceOffsetBytes = byteExtent(sourceOffset)
    val destinationOffsetBytes = byteExtent(destinationOffset)
    val copySizeBytes = byteExtent(elementCount)
    context.synchronizedLifecycle {
      requireOpen()
      source.requireOpen()
      stream.requireOpen()
      val sourceView =
        source.transferView(sourceOffsetBytes, copySizeBytes)
      copyResult(
        "CUDA asynchronous pinned host-to-device copy",
        stream.submitTracked(Vector(source, this)) { streamHandle =>
          backend.copyHostToDeviceAsync(
            context.nativeHandle,
            handle,
            destinationOffsetBytes,
            sourceView,
            streamHandle
          )
        }
      )
    }

  def copyToArray()(using
      codec: CudaHostCodec[T]
  ): Either[CudaDriverFailure, Array[T]] =
    requireHostCodec(codec)
    requireWholeBufferHostCopy()
    val destination = ByteBuffer
      .allocateDirect(sizeBytes.toInt)
      .order(ByteOrder.nativeOrder())
    context.synchronizedLifecycle {
      requireOpen()
      requireIdle()
      val status = backend.copyDeviceToHost(
        context.nativeHandle,
        handle,
        0L,
        destination
      )
      if status.succeeded then
        Right(codec.decode(destination, elementCount))
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA device-to-host copy",
            status
          )
        )
    }

  def copyTo(
      destination: CudaPinnedBuffer[T]
  ): Either[CudaDriverFailure, Unit] =
    requirePinnedBuffer(destination)
    requireMatchingPinnedBufferSize(destination)
    copyTo(
      destination = destination,
      sourceOffset = 0,
      destinationOffset = 0,
      elementCount = elementCount
    )

  def copyTo(
      destination: CudaPinnedBuffer[T],
      sourceOffset: Int,
      destinationOffset: Int,
      elementCount: Int
  ): Either[CudaDriverFailure, Unit] =
    requirePinnedBuffer(destination)
    requireElementRange(
      sourceOffset,
      elementCount,
      this.elementCount,
      "device source"
    )
    requireElementRange(
      destinationOffset,
      elementCount,
      destination.elementCount,
      "pinned destination"
    )
    val sourceOffsetBytes = byteExtent(sourceOffset)
    val destinationOffsetBytes = byteExtent(destinationOffset)
    val copySizeBytes = byteExtent(elementCount)
    context.synchronizedLifecycle {
      requireOpen()
      destination.requireOpen()
      requireIdle()
      destination.requireIdle()
      copyResult(
        "CUDA device-to-pinned-host copy",
        backend.copyDeviceToHost(
          context.nativeHandle,
          handle,
          sourceOffsetBytes,
          destination.transferView(
            destinationOffsetBytes,
            copySizeBytes
          )
        )
      )
    }

  def copyToAsync(
      destination: CudaPinnedBuffer[T],
      stream: CudaStream
  ): Either[CudaDriverFailure, Unit] =
    requirePinnedBuffer(destination)
    requireMatchingPinnedBufferSize(destination)
    copyToAsync(
      destination = destination,
      sourceOffset = 0,
      destinationOffset = 0,
      elementCount = elementCount,
      stream = stream
    )

  def copyToAsync(
      destination: CudaPinnedBuffer[T],
      sourceOffset: Int,
      destinationOffset: Int,
      elementCount: Int,
      stream: CudaStream
  ): Either[CudaDriverFailure, Unit] =
    requirePinnedBuffer(destination)
    requireStream(stream)
    requireElementRange(
      sourceOffset,
      elementCount,
      this.elementCount,
      "device source"
    )
    requireElementRange(
      destinationOffset,
      elementCount,
      destination.elementCount,
      "pinned destination"
    )
    val sourceOffsetBytes = byteExtent(sourceOffset)
    val destinationOffsetBytes = byteExtent(destinationOffset)
    val copySizeBytes = byteExtent(elementCount)
    context.synchronizedLifecycle {
      requireOpen()
      destination.requireOpen()
      stream.requireOpen()
      val destinationView = destination.transferView(
        destinationOffsetBytes,
        copySizeBytes
      )
      copyResult(
        "CUDA asynchronous device-to-pinned-host copy",
        stream.submitTracked(Vector(this, destination)) { streamHandle =>
          backend.copyDeviceToHostAsync(
            context.nativeHandle,
            handle,
            sourceOffsetBytes,
            destinationView,
            streamHandle
          )
        }
      )
    }

  override def close(): Unit =
    context.synchronizedLifecycle {
      if !lifetime.isReleased then
        context.requireOpen()
        lifetime.requestClose() match
          case Left(failure) => throw CudaDriverException(failure)
          case Right(()) => ()
    }

  override private[flight4s] def deviceAddress: DeviceAddress =
    context.synchronizedLifecycle {
      requireOpen()
      DeviceAddress.fromRaw(handle)
    }

  private def requireOpen(): Unit =
    context.requireOpen()
    if !lifetime.isOpen then
      throw IllegalStateException("CUDA device buffer is closed")

  private def requireIdle(): Unit =
    lifetime.requireIdle("CUDA device buffer")

  override private[cuda] def acquireInFlight()
      : InFlightLease[CudaDriverFailure] =
    requireOpen()
    lifetime.acquire()

  private def requireHostCodec(codec: CudaHostCodec[T]): Unit =
    require(
      codec.cudaType eq valueType,
      s"host codec ${codec.cudaType.cudaName} does not match " +
        s"device buffer type ${valueType.cudaName}"
    )

  private def requireWholeBufferHostCopy(): Unit =
    require(
      sizeBytes <= Int.MaxValue,
      s"whole-buffer host copies require at most ${Int.MaxValue} bytes: " +
        sizeBytes
    )

  private def requirePinnedBuffer(buffer: CudaPinnedBuffer[T]): Unit =
    require(
      buffer.context eq context,
      "CUDA pinned and device buffers must belong to the same context"
    )
    require(
      buffer.valueType eq valueType,
      s"pinned buffer type ${buffer.valueType.cudaName} does not match " +
        s"device buffer type ${valueType.cudaName}"
    )

  private def requireMatchingPinnedBufferSize(
      buffer: CudaPinnedBuffer[T]
  ): Unit =
    require(
      buffer.elementCount == elementCount &&
        buffer.sizeBytes == sizeBytes,
      "CUDA pinned and device buffer sizes must match"
    )

  private def requireStream(stream: CudaStream): Unit =
    require(
      stream.context eq context,
      "CUDA stream and buffers must belong to the same context"
    )

  private def requireElementRange(
      offset: Int,
      count: Int,
      capacity: Int,
      description: String
  ): Unit =
    require(offset >= 0, s"$description offset must not be negative: $offset")
    require(count > 0, s"$description element count must be positive: $count")
    val end = Math.addExact(offset.toLong, count.toLong)
    require(
      end <= capacity.toLong,
      s"$description range [$offset, $end) exceeds element count $capacity"
    )

  private def byteExtent(elements: Int): Long =
    Math.multiplyExact(elements.toLong, valueType.sizeBytes.toLong)

  private def copyResult(
      operation: String,
      status: NativeCudaDriverStatus
  ): Either[CudaDriverFailure, Unit] =
    if status.succeeded then Right(())
    else Left(CudaDriverFailure.fromStatus(operation, status))

private final class SharedCudaModule(
    context: CudaContext,
    key: PtxModuleKey,
    val handle: Long,
    backend: CudaDriverBackend
):
  private var wrapperCount = 0
  private var unloaded = false

  def retain(): Unit =
    require(!unloaded, "cannot retain an unloaded CUDA module")
    wrapperCount += 1

  def release(): Either[CudaDriverFailure, Unit] =
    require(wrapperCount > 0, "CUDA module wrapper count underflow")
    if wrapperCount > 1 then
      wrapperCount -= 1
      Right(())
    else
      val status = backend.unloadModule(context.nativeHandle, handle)
      if status.succeeded then
        wrapperCount = 0
        unloaded = true
        context.removeNativeModule(key, this)
        Right(())
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA module unload",
            status
          )
        )

final class CudaModule private[cuda] (
    val context: CudaContext,
    val artifact: NvrtcArtifact,
    private val shared: SharedCudaModule,
    private val backend: CudaDriverBackend
) extends AutoCloseable,
      CudaInFlightResource:
  private val lifetime = InFlightResourceState[CudaDriverFailure] { () =>
    shared.release().map { _ =>
      context.unregister(this)
    }
  }

  def isOpen: Boolean =
    context.synchronizedLifecycle(lifetime.isOpen && context.isOpen)

  def function[Args <: Tuple](
      raw: RawCudaKernel[Args]
  ): Either[CudaDriverFailure, CudaFunction[Args]] =
    context.synchronizedLifecycle {
      requireOpen()
      val owned = artifact.input match
        case input: NvrtcCompilationInput.Raw[?] => input.definition eq raw
        case _: NvrtcCompilationInput.Generated => false
      require(
        owned,
        s"raw kernel ${raw.entryPoint} does not belong to this CUDA module artifact"
      )
      resolveFunction(
        NvrtcKernelInput(raw.entryPoint, raw.signature, raw.launchRequirements),
        Some(raw)
      )
    }

  def function[Args <: Tuple](
      generated: GeneratedKernel[Args]
  ): Either[CudaDriverFailure, CudaFunction[Args]] =
    context.synchronizedLifecycle {
      requireOpen()
      require(
        owns(generated),
        s"kernel ${generated.name} does not belong to this CUDA module artifact"
      )
      val canonical = artifact.input.kernels.find(_.entryPoint == generated.name).get
      resolveFunction(
        NvrtcKernelInput(generated.name, generated.signature, canonical.launchRequirements),
        None
      )
    }

  private def resolveFunction[Args <: Tuple](
      kernel: NvrtcKernelInput[Args],
      rawDefinition: Option[RawCudaKernel[Args]]
  ): Either[CudaDriverFailure, CudaFunction[Args]] =
    val functionResult = backend.resolveFunction(
      context.nativeHandle,
      shared.handle,
      kernel.entryPoint
    )
    if functionResult.status.succeeded then
      require(
        functionResult.handle != 0L,
        "successful CUDA function lookup returned a null handle"
      )
      val attributesResult = backend.queryFunctionAttributes(
        context.nativeHandle,
        functionResult.handle
      )
      if attributesResult.status.succeeded then
        Right(
          new CudaFunction(
            this,
            kernel,
            rawDefinition,
            functionResult.handle,
            CudaFunctionAttributes(
              maxThreadsPerBlock = attributesResult.maxThreadsPerBlock,
              staticSharedMemoryBytes = attributesResult.staticSharedMemoryBytes,
              constantMemoryBytes = attributesResult.constantMemoryBytes,
              localMemoryBytes = attributesResult.localMemoryBytes,
              registersPerThread = attributesResult.registersPerThread
            )
          )
        )
      else
        Left(
          CudaDriverFailure.fromStatus(
            s"CUDA function attribute query for ${kernel.entryPoint}",
            attributesResult.status
          )
        )
    else
      Left(
        CudaDriverFailure.fromStatus(
          s"CUDA function lookup for ${kernel.entryPoint}",
          functionResult.status
        )
      )

  override def close(): Unit =
    context.synchronizedLifecycle {
      if !lifetime.isReleased then
        context.requireOpen()
        lifetime.requestClose() match
          case Left(failure) => throw CudaDriverException(failure)
          case Right(()) => ()
    }

  private[cuda] def requireOpen(): Unit =
    context.requireOpen()
    if !lifetime.isOpen then
      throw IllegalStateException("CUDA module is closed")

  override private[cuda] def acquireInFlight()
      : InFlightLease[CudaDriverFailure] =
    requireOpen()
    lifetime.acquire()

  private[cuda] def submit(
      functionHandle: Long,
      streamHandle: Long,
      request: NativeLaunchRequest
  ): NativeCudaDriverStatus =
    backend.launchKernel(
      contextHandle = context.nativeHandle,
      functionHandle = functionHandle,
      streamHandle = streamHandle,
      request = request
    )

  private def owns[Args <: Tuple](
      generated: GeneratedKernel[Args]
  ): Boolean =
    artifact.input match
      case input: NvrtcCompilationInput.Generated =>
        input.module.kernels.exists { candidate =>
          candidate.name == generated.name &&
          (candidate.signature eq generated.signature)
        }
      case _: NvrtcCompilationInput.Raw[?] => false

final class CudaFunction[Args <: Tuple] private[cuda] (
    val module: CudaModule,
    val kernel: NvrtcKernelInput[Args],
    private val rawDefinition: Option[RawCudaKernel[Args]],
    private val handle: Long,
    val attributes: CudaFunctionAttributes
):
  def name: String = kernel.entryPoint
  def signature: KernelSignature[Args] = kernel.signature
  def provenance: NvrtcSourceProvenance = module.artifact.provenance
  def isValid: Boolean = module.isOpen

  def launch(
      invocation: KernelInvocation[Args],
      config: LaunchConfig
  ): Either[CudaLaunchFailure, Unit] =
    launchOn(invocation, config, None)

  def launch(
      invocation: KernelInvocation[Args],
      config: LaunchConfig,
      stream: CudaStream
  ): Either[CudaLaunchFailure, Unit] =
    launchOn(invocation, config, Some(stream))

  def launch(
      invocation: RawCudaInvocation[Args],
      config: LaunchConfig
  ): Either[CudaLaunchFailure, Unit] =
    launchOn(invocation, config, None)

  def launch(
      invocation: RawCudaInvocation[Args],
      config: LaunchConfig,
      stream: CudaStream
  ): Either[CudaLaunchFailure, Unit] =
    launchOn(invocation, config, Some(stream))

  private def launchOn(
      invocation: KernelInvocation[Args] | RawCudaInvocation[Args],
      config: LaunchConfig,
      stream: Option[CudaStream]
  ): Either[CudaLaunchFailure, Unit] =
    module.context.synchronizedLifecycle {
      module.requireOpen()
      val arguments = invocation match
        case value: KernelInvocation[Args] => value.arguments
        case value: RawCudaInvocation[Args] => value.arguments
      for
        _ <- validateInvocation(invocation)
        _ <- validateBlockShape(config)
        _ <- validateDynamicSharedMemory(config)
        streamHandle <- validateStream(stream)
        _ <- validateBuffers(arguments)
        request = NativeLaunchRequest.materialize(
          config,
          NativeArgumentStorage.materialize(signature.pack(arguments))
        )
        status = stream match
          case Some(value) =>
            value.submitTracked(launchResources(arguments)) { trackedStreamHandle =>
              module.submit(handle, trackedStreamHandle, request)
            }
          case None =>
            module.context.submitDefaultTracked(launchResources(arguments)) {
              module.submit(handle, streamHandle, request)
            }
        _ <-
          if status.succeeded then Right(())
          else
            Left(
              CudaLaunchFailure.Driver(
                CudaDriverFailure.fromStatus(s"CUDA kernel launch for $name", status)
              )
            )
      yield ()
    }

  private[flight4s] def nativeHandle: Long =
    module.context.synchronizedLifecycle {
      module.requireOpen()
      handle
    }

  private def validateInvocation(
      invocation: KernelInvocation[Args] | RawCudaInvocation[Args]
  ): Either[CudaLaunchFailure, Unit] =
    val (matches, actualName) = invocation match
      case value: KernelInvocation[Args] =>
        (
          rawDefinition.isEmpty && value.kernel.name == name &&
            (value.kernel.signature eq signature),
          value.kernel.name
        )
      case value: RawCudaInvocation[Args] =>
        (rawDefinition.exists(_ eq value.kernel), value.kernel.entryPoint)
    if matches then Right(())
    else
      Left(
        CudaLaunchFailure.InvocationMismatch(
          expectedKernel = name,
          actualKernel = actualName
        )
      )

  private def validateBlockShape(config: LaunchConfig): Either[CudaLaunchFailure, Unit] =
    kernel.launchRequirements.requiredBlock match
      case Some(required) if required != config.block =>
        Left(CudaLaunchFailure.BlockShapeMismatch(name, required, config.block))
      case _ => Right(())

  private def validateDynamicSharedMemory(
      config: LaunchConfig
  ): Either[CudaLaunchFailure, Unit] =
    kernel.launchRequirements.dynamicSharedMemory match
      case None => Right(())
      case Some(requirement)
          if config.dynamicSharedMemoryBytes == 0 =>
        Left(
          CudaLaunchFailure.DynamicSharedMemoryRequired(
            kernelName = name,
            elementSizeBytes = requirement.elementSizeBytes
          )
        )
      case Some(requirement)
          if config.dynamicSharedMemoryBytes <
              requirement.elementSizeBytes ||
            config.dynamicSharedMemoryBytes %
                requirement.elementSizeBytes != 0 ||
            config.dynamicSharedMemoryBytes %
                requirement.elementAlignmentBytes != 0 =>
        Left(
          CudaLaunchFailure.InvalidDynamicSharedMemorySize(
            kernelName = name,
            configuredBytes = config.dynamicSharedMemoryBytes,
            elementSizeBytes = requirement.elementSizeBytes,
            elementAlignmentBytes = requirement.elementAlignmentBytes
          )
        )
      case Some(_) => Right(())

  private def validateStream(
      stream: Option[CudaStream]
  ): Either[CudaLaunchFailure, Long] =
    stream match
      case None => Right(0L)
      case Some(value) if !(value.context eq module.context) =>
        Left(CudaLaunchFailure.StreamContextMismatch(name))
      case Some(value) =>
        value.requireOpen()
        Right(value.nativeHandle)

  private def launchResources(
      arguments: Args
  ): Vector[CudaInFlightResource] =
    val buffers = arguments.productIterator.collect {
      case resource: CudaInFlightResource => resource
    }.toVector
    module +: buffers

  private def validateBuffers(arguments: Args): Either[CudaLaunchFailure, Unit] =
    arguments.productIterator.zipWithIndex.collectFirst {
      case (buffer: CudaDeviceBuffer[?], index)
          if !(buffer.context eq module.context) =>
        CudaLaunchFailure.BufferContextMismatch(name, index)
    }.toLeft(())

object CudaDriverFailure:
  private[cuda] def fromStatus(
      operation: String,
      status: NativeCudaDriverStatus
  ): CudaDriverFailure =
    CudaDriverFailure(
      operation = operation,
      resultCode = status.resultCode,
      resultName = status.resultName,
      resultDescription = status.resultDescription,
      infoLog = status.infoLog,
      errorLog = status.errorLog
    )
