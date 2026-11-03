package flight4s.runtime.cuda.internal

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

import flight4s.core.abi.NativeLaunchRequest

private[flight4s] final case class NativeCudaDriverStatus(
    resultCode: Int,
    resultName: String,
    resultDescription: String,
    infoLog: String = "",
    errorLog: String = ""
):
  def succeeded: Boolean = resultCode == 0

private[flight4s] final case class NativeCudaContextRetainResult(
    handle: Long,
    deviceOrdinal: Int,
    computeCapabilityMajor: Int,
    computeCapabilityMinor: Int,
    status: NativeCudaDriverStatus
)

private[flight4s] final case class NativeCudaResourceResult(
    handle: Long,
    status: NativeCudaDriverStatus
)

private[flight4s] final case class NativeCudaFunctionAttributesQuery(
    maxThreadsPerBlock: Int,
    staticSharedMemoryBytes: Int,
    constantMemoryBytes: Int,
    localMemoryBytes: Int,
    registersPerThread: Int,
    status: NativeCudaDriverStatus
)

private[flight4s] final case class NativeCudaPinnedAllocationResult(
    handle: Long,
    storage: ByteBuffer,
    status: NativeCudaDriverStatus
)

private[flight4s] final case class NativeCudaEventQuery(
    complete: Boolean,
    status: NativeCudaDriverStatus
)

private[flight4s] trait CudaDriverBackend:
  def retainPrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaContextRetainResult

  def releasePrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaDriverStatus

  def synchronizeContext(
      contextHandle: Long
  ): NativeCudaDriverStatus

  def loadPtx(
      contextHandle: Long,
      ptx: IArray[Byte]
  ): NativeCudaResourceResult

  def unloadModule(
      contextHandle: Long,
      moduleHandle: Long
  ): NativeCudaDriverStatus

  def resolveFunction(
      contextHandle: Long,
      moduleHandle: Long,
      functionName: String
  ): NativeCudaResourceResult

  def queryFunctionAttributes(
      contextHandle: Long,
      functionHandle: Long
  ): NativeCudaFunctionAttributesQuery

  def launchKernel(
      contextHandle: Long,
      functionHandle: Long,
      streamHandle: Long,
      request: NativeLaunchRequest
  ): NativeCudaDriverStatus

  def createStream(
      contextHandle: Long,
      flags: Int
  ): NativeCudaResourceResult

  def destroyStream(
      contextHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus

  def synchronizeStream(
      contextHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus

  def createEvent(
      contextHandle: Long,
      flags: Int
  ): NativeCudaResourceResult

  def destroyEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus

  def recordEvent(
      contextHandle: Long,
      eventHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus

  def queryEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaEventQuery

  def synchronizeEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus

  def waitForEvent(
      contextHandle: Long,
      streamHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus

  def createBlasHandle(
      contextHandle: Long
  ): NativeCudaResourceResult

  def destroyBlasHandle(
      contextHandle: Long,
      blasHandle: Long
  ): NativeCudaDriverStatus

  def executeGemmF32(
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
  ): NativeCudaDriverStatus

  def allocatePinnedMemory(
      contextHandle: Long,
      sizeBytes: Long
  ): NativeCudaPinnedAllocationResult

  def freePinnedMemory(
      contextHandle: Long,
      hostAddress: Long
  ): NativeCudaDriverStatus

  def allocateDeviceMemory(
      contextHandle: Long,
      sizeBytes: Long
  ): NativeCudaResourceResult

  def freeDeviceMemory(
      contextHandle: Long,
      deviceAddress: Long
  ): NativeCudaDriverStatus

  def copyHostToDevice(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      source: ByteBuffer
  ): NativeCudaDriverStatus

  def copyHostToDeviceAsync(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      source: ByteBuffer,
      streamHandle: Long
  ): NativeCudaDriverStatus

  def copyDeviceToHost(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      destination: ByteBuffer
  ): NativeCudaDriverStatus

  def copyDeviceToHostAsync(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      destination: ByteBuffer,
      streamHandle: Long
  ): NativeCudaDriverStatus

private[flight4s] object NativeCudaDriver extends CudaDriverBackend:
  override def retainPrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaContextRetainResult =
    val result =
      CudaNativeBindings.retainPrimaryContext(deviceOrdinal)
    NativeCudaContextRetainResult(
      handle = result.handle(),
      deviceOrdinal = result.deviceOrdinal(),
      computeCapabilityMajor = result.computeCapabilityMajor(),
      computeCapabilityMinor = result.computeCapabilityMinor(),
      status = NativeCudaDriverStatus(
        resultCode = result.resultCode(),
        resultName = decode(result.resultNameUtf8()),
        resultDescription = decode(result.resultDescriptionUtf8())
      )
    )

  override def releasePrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaDriverStatus =
    status(CudaNativeBindings.releasePrimaryContext(deviceOrdinal))

  override def synchronizeContext(
      contextHandle: Long
  ): NativeCudaDriverStatus =
    status(CudaNativeBindings.synchronizeContext(contextHandle))

  override def loadPtx(
      contextHandle: Long,
      ptx: IArray[Byte]
  ): NativeCudaResourceResult =
    resource(
      CudaNativeBindings.loadPtx(
        contextHandle,
        ptx.asInstanceOf[Array[Byte]]
      )
    )

  override def unloadModule(
      contextHandle: Long,
      moduleHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.unloadModule(contextHandle, moduleHandle)
    )

  override def resolveFunction(
      contextHandle: Long,
      moduleHandle: Long,
      functionName: String
  ): NativeCudaResourceResult =
    resource(
      CudaNativeBindings.resolveFunction(
        contextHandle,
        moduleHandle,
        functionName.getBytes(StandardCharsets.UTF_8)
      )
    )

  override def queryFunctionAttributes(
      contextHandle: Long,
      functionHandle: Long
  ): NativeCudaFunctionAttributesQuery =
    val result = CudaNativeBindings.queryFunctionAttributes(
      contextHandle,
      functionHandle
    )
    NativeCudaFunctionAttributesQuery(
      maxThreadsPerBlock = result.maxThreadsPerBlock(),
      staticSharedMemoryBytes = result.staticSharedMemoryBytes(),
      constantMemoryBytes = result.constantMemoryBytes(),
      localMemoryBytes = result.localMemoryBytes(),
      registersPerThread = result.registersPerThread(),
      status = status(result)
    )

  override def launchKernel(
      contextHandle: Long,
      functionHandle: Long,
      streamHandle: Long,
      request: NativeLaunchRequest
  ): NativeCudaDriverStatus =
    status(
      NativeCudaLauncher.launch(
        contextHandle,
        functionHandle,
        streamHandle,
        request
      )
    )

  override def createStream(
      contextHandle: Long,
      flags: Int
  ): NativeCudaResourceResult =
    resource(CudaNativeBindings.createStream(contextHandle, flags))

  override def destroyStream(
      contextHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    status(CudaNativeBindings.destroyStream(contextHandle, streamHandle))

  override def synchronizeStream(
      contextHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.synchronizeStream(
        contextHandle,
        streamHandle
      )
    )

  override def createEvent(
      contextHandle: Long,
      flags: Int
  ): NativeCudaResourceResult =
    resource(CudaNativeBindings.createEvent(contextHandle, flags))

  override def destroyEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus =
    status(CudaNativeBindings.destroyEvent(contextHandle, eventHandle))

  override def recordEvent(
      contextHandle: Long,
      eventHandle: Long,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.recordEvent(
        contextHandle,
        eventHandle,
        streamHandle
      )
    )

  override def queryEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaEventQuery =
    val result = CudaNativeBindings.queryEvent(
      contextHandle,
      eventHandle
    )
    NativeCudaEventQuery(
      complete = result.complete(),
      status = status(result)
    )

  override def synchronizeEvent(
      contextHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.synchronizeEvent(
        contextHandle,
        eventHandle
      )
    )

  override def waitForEvent(
      contextHandle: Long,
      streamHandle: Long,
      eventHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.waitForEvent(
        contextHandle,
        streamHandle,
        eventHandle
      )
    )

  override def createBlasHandle(
      contextHandle: Long
  ): NativeCudaResourceResult =
    resource(CudaNativeBindings.createBlasHandle(contextHandle))

  override def destroyBlasHandle(
      contextHandle: Long,
      blasHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.destroyBlasHandle(contextHandle, blasHandle)
    )

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
    status(
      CudaNativeBindings.executeGemmF32(
        contextHandle,
        blasHandle,
        streamHandle,
        m,
        n,
        k,
        alpha,
        beta,
        aAddress,
        bAddress,
        cAddress,
        workspaceAddress,
        workspaceBytes
      )
    )

  override def allocatePinnedMemory(
      contextHandle: Long,
      sizeBytes: Long
  ): NativeCudaPinnedAllocationResult =
    val result = CudaNativeBindings.allocatePinnedMemory(
      contextHandle,
      sizeBytes
    )
    NativeCudaPinnedAllocationResult(
      handle = result.handle(),
      storage = result.storage(),
      status = status(result)
    )

  override def freePinnedMemory(
      contextHandle: Long,
      hostAddress: Long
  ): NativeCudaDriverStatus =
    status(CudaNativeBindings.freePinnedMemory(contextHandle, hostAddress))

  override def allocateDeviceMemory(
      contextHandle: Long,
      sizeBytes: Long
  ): NativeCudaResourceResult =
    resource(
      CudaNativeBindings.allocateDeviceMemory(
        contextHandle,
        sizeBytes
      )
    )

  override def freeDeviceMemory(
      contextHandle: Long,
      deviceAddress: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.freeDeviceMemory(
        contextHandle,
        deviceAddress
      )
    )

  override def copyHostToDevice(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      source: ByteBuffer
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.copyHostToDevice(
        contextHandle,
        deviceAddress,
        deviceOffsetBytes,
        source
      )
    )

  override def copyHostToDeviceAsync(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      source: ByteBuffer,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.copyHostToDeviceAsync(
        contextHandle,
        deviceAddress,
        deviceOffsetBytes,
        source,
        streamHandle
      )
    )

  override def copyDeviceToHost(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      destination: ByteBuffer
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.copyDeviceToHost(
        contextHandle,
        deviceAddress,
        deviceOffsetBytes,
        destination
      )
    )

  override def copyDeviceToHostAsync(
      contextHandle: Long,
      deviceAddress: Long,
      deviceOffsetBytes: Long,
      destination: ByteBuffer,
      streamHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.copyDeviceToHostAsync(
        contextHandle,
        deviceAddress,
        deviceOffsetBytes,
        destination,
        streamHandle
      )
    )

  private def resource(
      result: NativeCudaDriverResult
  ): NativeCudaResourceResult =
    NativeCudaResourceResult(
      handle = result.handle(),
      status = status(result)
    )

  private def status(
      result: NativeCudaDriverResult
  ): NativeCudaDriverStatus =
    NativeCudaDriverStatus(
      resultCode = result.resultCode(),
      resultName = decode(result.resultNameUtf8()),
      resultDescription = decode(result.resultDescriptionUtf8()),
      infoLog = decode(result.infoLogUtf8()),
      errorLog = decode(result.errorLogUtf8())
    )

  private def status(
      result: NativeCudaPinnedMemoryResult
  ): NativeCudaDriverStatus =
    NativeCudaDriverStatus(
      resultCode = result.resultCode(),
      resultName = decode(result.resultNameUtf8()),
      resultDescription = decode(result.resultDescriptionUtf8()),
      infoLog = decode(result.infoLogUtf8()),
      errorLog = decode(result.errorLogUtf8())
    )

  private def status(
      result: NativeCudaFunctionAttributesResult
  ): NativeCudaDriverStatus =
    NativeCudaDriverStatus(
      resultCode = result.resultCode(),
      resultName = decode(result.resultNameUtf8()),
      resultDescription = decode(result.resultDescriptionUtf8()),
      infoLog = decode(result.infoLogUtf8()),
      errorLog = decode(result.errorLogUtf8())
    )

  private def status(
      result: flight4s.runtime.cuda.internal.NativeCudaEventQueryResult
  ): NativeCudaDriverStatus =
    NativeCudaDriverStatus(
      resultCode = result.resultCode(),
      resultName = decode(result.resultNameUtf8()),
      resultDescription = decode(result.resultDescriptionUtf8()),
      infoLog = decode(result.infoLogUtf8()),
      errorLog = decode(result.errorLogUtf8())
    )

  private def decode(bytes: Array[Byte]): String =
    String(bytes, StandardCharsets.UTF_8)
