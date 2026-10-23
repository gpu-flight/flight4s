package flight4s.runtime.cuda

import java.nio.{ByteBuffer, ByteOrder}
import scala.collection.mutable.{ArrayBuffer, HashMap}
import flight4s.runtime.cuda.internal.*

private[cuda] object RecordingBackend:
  val successStatus: NativeCudaDriverStatus =
    NativeCudaDriverStatus(
      resultCode = 0,
      resultName = "CUDA_SUCCESS",
      resultDescription = "no error"
    )

  val defaultFunctionAttributes: CudaFunctionAttributes = CudaFunctionAttributes(
    maxThreadsPerBlock = 1024,
    staticSharedMemoryBytes = 0,
    constantMemoryBytes = 0,
    localMemoryBytes = 0,
    registersPerThread = 32
  )

/** Shared fake driver backend for runtime suites; records every native call. */
private[cuda] final class RecordingBackend extends CudaDriverBackend:
  import RecordingBackend.{successStatus, defaultFunctionAttributes}

  val events: ArrayBuffer[String] = ArrayBuffer.empty
  var lastLaunch = Option.empty[flight4s.core.abi.NativeLaunchRequest]
  var loadStatus: NativeCudaDriverStatus = successStatus
  var unloadStatus: NativeCudaDriverStatus = successStatus
  var resolveStatus: NativeCudaDriverStatus = successStatus
  var functionAttributesStatus: NativeCudaDriverStatus = successStatus
  var releaseStatus: NativeCudaDriverStatus = successStatus
  var synchronizeContextStatus: NativeCudaDriverStatus = successStatus
  var launchStatus: NativeCudaDriverStatus = successStatus
  var createStreamStatus: NativeCudaDriverStatus = successStatus
  var destroyStreamStatus: NativeCudaDriverStatus = successStatus
  var synchronizeStreamStatus: NativeCudaDriverStatus = successStatus
  var createEventStatus: NativeCudaDriverStatus = successStatus
  var destroyEventStatus: NativeCudaDriverStatus = successStatus
  var recordEventStatus: NativeCudaDriverStatus = successStatus
  var queryEventStatus: NativeCudaDriverStatus = successStatus
  var queryEventComplete = false
  var synchronizeEventStatus: NativeCudaDriverStatus = successStatus
  var waitForEventStatus: NativeCudaDriverStatus = successStatus
  var createBlasStatus: NativeCudaDriverStatus = successStatus
  var destroyBlasStatus: NativeCudaDriverStatus = successStatus
  var gemmStatus: NativeCudaDriverStatus = successStatus
  var allocatePinnedStatus: NativeCudaDriverStatus = successStatus
  var freePinnedStatus: NativeCudaDriverStatus = successStatus
  var allocateStatus: NativeCudaDriverStatus = successStatus
  var freeStatus: NativeCudaDriverStatus = successStatus
  var hostToDeviceStatus: NativeCudaDriverStatus = successStatus
  var deviceToHostStatus: NativeCudaDriverStatus = successStatus
  private var nextModuleHandle = 200L
  private var nextStreamHandle = 400L
  private var nextEventHandle = 500L
  private var nextPinnedAddress = 2000L
  private var nextDeviceAddress = 1000L
  private val memory = HashMap.empty[Long, Array[Byte]]
  private val pinnedMemory = HashMap.empty[Long, ByteBuffer]

  override def retainPrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaContextRetainResult =
    events += s"retain:$deviceOrdinal"
    NativeCudaContextRetainResult(
      handle = 100L,
      deviceOrdinal = deviceOrdinal,
      computeCapabilityMajor = 8,
      computeCapabilityMinor = 9,
      status = successStatus
    )

  override def releasePrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaDriverStatus =
    events += s"release:$deviceOrdinal"
    releaseStatus

  override def synchronizeContext(
      contextHandle: Long
  ): NativeCudaDriverStatus =
    events += s"synchronizeContext:$contextHandle"
    synchronizeContextStatus

  override def loadPtx(
      contextHandle: Long,
      ptx: IArray[Byte]
  ): NativeCudaResourceResult =
    events += s"load:$contextHandle"
    val handle = nextModuleHandle
    nextModuleHandle += 1
    NativeCudaResourceResult(
      handle = if loadStatus.succeeded then handle else 0L,
      status = loadStatus
    )

  override def unloadModule(
      contextHandle: Long,
      moduleHandle: Long
  ): NativeCudaDriverStatus =
    events += s"unload:$contextHandle:$moduleHandle"
    unloadStatus

  override def resolveFunction(
      contextHandle: Long,
      moduleHandle: Long,
      functionName: String
  ): NativeCudaResourceResult =
    events +=
      s"resolve:$contextHandle:$moduleHandle:$functionName"
    NativeCudaResourceResult(
      handle = if resolveStatus.succeeded then 300L else 0L,
      status = resolveStatus
    )

  override def queryFunctionAttributes(
      contextHandle: Long,
      functionHandle: Long
  ): NativeCudaFunctionAttributesQuery =
    events += s"attributes:$contextHandle:$functionHandle"
    NativeCudaFunctionAttributesQuery(
      maxThreadsPerBlock = defaultFunctionAttributes.maxThreadsPerBlock,
      staticSharedMemoryBytes =
        defaultFunctionAttributes.staticSharedMemoryBytes,
      constantMemoryBytes = defaultFunctionAttributes.constantMemoryBytes,
      localMemoryBytes = defaultFunctionAttributes.localMemoryBytes,
      registersPerThread = defaultFunctionAttributes.registersPerThread,
      status = functionAttributesStatus
    )

  override def launchKernel(
      contextHandle: Long,
      functionHandle: Long,
      streamHandle: Long,
      request: flight4s.core.abi.NativeLaunchRequest
  ): NativeCudaDriverStatus =
    lastLaunch = Some(request)
    events +=
      s"launch:$contextHandle:$functionHandle:$streamHandle:" +
        s"${request.gridX}:${request.blockX}"
    launchStatus

  override def createStream(
      contextHandle: Long,
      flags: Int
  ): NativeCudaResourceResult =
    events += s"createStream:$contextHandle:$flags"
    val handle = nextStreamHandle
    nextStreamHandle += 1
    NativeCudaResourceResult(
      handle = if createStreamStatus.succeeded then handle else 0L,
      status = createStreamStatus
    )

  override def destroyStream(
      contextHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    events += s"destroyStream:$contextHandle:$streamHandle"
    destroyStreamStatus

  override def synchronizeStream(
      contextHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    events += s"synchronizeStream:$contextHandle:$streamHandle"
    synchronizeStreamStatus

  override def createEvent(
      contextHandle: Long,
      flags: Int
  ): NativeCudaResourceResult =
    events += s"createEvent:$contextHandle:$flags"
    val handle = nextEventHandle
    nextEventHandle += 1
    NativeCudaResourceResult(
      handle = if createEventStatus.succeeded then handle else 0L,
      status = createEventStatus
    )

  override def destroyEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus =
    events += s"destroyEvent:$contextHandle:$eventHandle"
    destroyEventStatus

  override def recordEvent(
      contextHandle: Long,
      eventHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    events += s"recordEvent:$contextHandle:$eventHandle:$streamHandle"
    recordEventStatus

  override def queryEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaEventQuery =
    events += s"queryEvent:$contextHandle:$eventHandle"
    NativeCudaEventQuery(
      complete = queryEventComplete,
      status = queryEventStatus
    )

  override def synchronizeEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus =
    events += s"synchronizeEvent:$contextHandle:$eventHandle"
    synchronizeEventStatus

  override def waitForEvent(
      contextHandle: Long,
      streamHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus =
    events += s"waitForEvent:$contextHandle:$streamHandle:$eventHandle"
    waitForEventStatus

  private var nextBlasHandle = 600L

  override def createBlasHandle(
      contextHandle: Long
  ): NativeCudaResourceResult =
    events += s"createBlas:$contextHandle"
    val handle = nextBlasHandle
    nextBlasHandle += 1
    NativeCudaResourceResult(
      handle = if createBlasStatus.succeeded then handle else 0L,
      status = createBlasStatus
    )

  override def destroyBlasHandle(
      contextHandle: Long,
      blasHandle: Long
  ): NativeCudaDriverStatus =
    events += s"destroyBlas:$contextHandle:$blasHandle"
    destroyBlasStatus

  override def executeGemmF32(
      contextHandle: Long,
      blasHandle: Long,
      streamHandle: Long,
      m: Int,
      n: Int,
      k: Int,
      alpha: Float,
      beta: Float,
      aAddress: Long,
      bAddress: Long,
      cAddress: Long,
      workspaceAddress: Long,
      workspaceBytes: Long
  ): NativeCudaDriverStatus =
    events +=
      s"gemm:$contextHandle:$blasHandle:$streamHandle:$m:$n:$k:" +
        s"$alpha:$beta:$aAddress:$bAddress:$cAddress:$workspaceAddress:$workspaceBytes"
    gemmStatus

  override def allocatePinnedMemory(
      contextHandle: Long,
      sizeBytes: Long
  ): NativeCudaPinnedAllocationResult =
    events += s"allocatePinned:$contextHandle:$sizeBytes"
    val address = nextPinnedAddress
    nextPinnedAddress += 1
    val storage =
      if allocatePinnedStatus.succeeded then
        ByteBuffer
          .allocateDirect(sizeBytes.toInt)
          .order(ByteOrder.nativeOrder())
      else null
    if storage != null then pinnedMemory(address) = storage
    NativeCudaPinnedAllocationResult(
      handle = if allocatePinnedStatus.succeeded then address else 0L,
      storage = storage,
      status = allocatePinnedStatus
    )

  override def freePinnedMemory(
      contextHandle: Long,
      hostAddress: Long
  ): NativeCudaDriverStatus =
    events += s"freePinned:$contextHandle:$hostAddress"
    if freePinnedStatus.succeeded then pinnedMemory.remove(hostAddress)
    freePinnedStatus

  override def allocateDeviceMemory(
      contextHandle: Long,
      sizeBytes: Long
  ): NativeCudaResourceResult =
    events += s"allocate:$contextHandle:$sizeBytes"
    val address = nextDeviceAddress
    nextDeviceAddress += 1
    if allocateStatus.succeeded then
      memory(address) = new Array[Byte](sizeBytes.toInt)
    NativeCudaResourceResult(
      handle = if allocateStatus.succeeded then address else 0L,
      status = allocateStatus
    )

  override def freeDeviceMemory(
      contextHandle: Long,
      deviceAddress: Long
  ): NativeCudaDriverStatus =
    events += s"free:$contextHandle:$deviceAddress"
    if freeStatus.succeeded then memory.remove(deviceAddress)
    freeStatus

  override def copyHostToDevice(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      source: ByteBuffer
  ): NativeCudaDriverStatus =
    events +=
      s"copyHtoD:$contextHandle:$deviceAddress:" +
        s"$deviceOffsetBytes:${source.remaining()}"
    if hostToDeviceStatus.succeeded then
      val input = new Array[Byte](source.remaining())
      source.duplicate().get(input)
      Array.copy(
        input,
        0,
        memory(deviceAddress),
        Math.toIntExact(deviceOffsetBytes),
        input.length
      )
    hostToDeviceStatus

  override def copyHostToDeviceAsync(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      source: ByteBuffer,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    events +=
      s"copyHtoDAsync:$contextHandle:$deviceAddress:" +
        s"$deviceOffsetBytes:${source.remaining()}:$streamHandle"
    if hostToDeviceStatus.succeeded then
      val input = new Array[Byte](source.remaining())
      source.duplicate().get(input)
      Array.copy(
        input,
        0,
        memory(deviceAddress),
        Math.toIntExact(deviceOffsetBytes),
        input.length
      )
    hostToDeviceStatus

  override def copyDeviceToHost(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      destination: ByteBuffer
  ): NativeCudaDriverStatus =
    events +=
      s"copyDtoH:$contextHandle:$deviceAddress:" +
        s"$deviceOffsetBytes:${destination.remaining()}"
    if deviceToHostStatus.succeeded then
      destination.duplicate().put(
        memory(deviceAddress),
        Math.toIntExact(deviceOffsetBytes),
        destination.remaining()
      )
    deviceToHostStatus

  override def copyDeviceToHostAsync(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      destination: ByteBuffer,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    events +=
      s"copyDtoHAsync:$contextHandle:$deviceAddress:" +
        s"$deviceOffsetBytes:${destination.remaining()}:$streamHandle"
    if deviceToHostStatus.succeeded then
      destination.duplicate().put(
        memory(deviceAddress),
        Math.toIntExact(deviceOffsetBytes),
        destination.remaining()
      )
    deviceToHostStatus
