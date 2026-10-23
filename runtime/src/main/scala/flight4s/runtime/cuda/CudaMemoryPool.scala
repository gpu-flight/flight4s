package flight4s.runtime.cuda

import java.nio.{ByteBuffer, ByteOrder}
import flight4s.core.types.CudaType
import flight4s.runtime.cuda.internal.CudaDriverBackend

/** Pool limits: cached (idle) bytes per pool, and the largest pooled request. */
final case class MemoryPoolConfig(
    maxCachedBytes: Long = 256L * 1024 * 1024,
    maxBlockBytes: Long = 64L * 1024 * 1024
):
  require(maxCachedBytes >= 0L, "maxCachedBytes must not be negative")
  require(maxBlockBytes >= 256L, "maxBlockBytes must be at least 256 bytes")

final case class PoolStatistics(
    hits: Long,
    misses: Long,
    passThroughs: Long,
    cachedBytes: Long,
    liveBytes: Long,
    failedReleases: Long
)

object CudaMemoryPool:
  /** Loan-pattern pool: trims and closes on every exit path. */
  def scoped[A](context: CudaContext, config: MemoryPoolConfig = MemoryPoolConfig())(
      body: CudaMemoryPool => Either[CudaDriverFailure, A]
  ): Either[CudaDriverFailure, A] =
    context.createPool(config).flatMap { pool =>
      val result =
        try body(pool)
        catch
          case error: Throwable =>
            pool.close()
            throw error
      result match
        case Left(failure) =>
          pool.close()
          Left(failure)
        case Right(value) => pool.close().map(_ => value)
    }

/** Caches freed device and pinned blocks per power-of-two size class. Pooled
  * buffers keep their ordinary types and lease semantics; only close() changes
  * meaning, returning the block to the pool instead of freeing it. */
final class CudaMemoryPool private[cuda] (
    context: CudaContext,
    backend: CudaDriverBackend,
    config: MemoryPoolConfig
):
  private final class DeviceBlock(val handle: Long, val classBytes: Long)
  private final class PinnedBlock(val handle: Long, val classBytes: Long, val storage: ByteBuffer)

  private var deviceCached = Vector.empty[DeviceBlock]
  private var pinnedCached = Vector.empty[PinnedBlock]
  private var liveDevice = Vector.empty[DeviceBlock]
  private var livePinned = Vector.empty[PinnedBlock]
  private var closed = false
  private var hits = 0L
  private var misses = 0L
  private var passThroughs = 0L
  private var failedReleases = 0L
  private var cachedBytesCount = 0L
  private var liveBytesCount = 0L

  private[cuda] val contextResource: AutoCloseable = () => closeFromContext()

  def statistics: PoolStatistics = context.synchronizedLifecycle {
    PoolStatistics(hits, misses, passThroughs, cachedBytesCount, liveBytesCount, failedReleases)
  }

  def allocate[T](elementCount: Int)(using
      valueType: CudaType[T]
  ): Either[CudaDriverFailure, CudaDeviceBuffer[T]] =
    context.synchronizedLifecycle {
      requireOpenPool()
      require(
        elementCount > 0,
        s"pooled device buffer element count must be positive: $elementCount"
      )
      val sizeBytes = Math.multiplyExact(elementCount.toLong, valueType.sizeBytes.toLong)
      if sizeBytes > config.maxBlockBytes then
        passThroughs += 1
        context.allocate[T](elementCount)
      else
        val classBytes = roundUp(sizeBytes)
        takeDevice(classBytes).map(Right(_)).getOrElse {
          val result = backend.allocateDeviceMemory(context.nativeHandle, classBytes)
          if result.status.succeeded then
            require(result.handle != 0L, "successful CUDA device allocation returned a null address")
            misses += 1
            Right(new DeviceBlock(result.handle, classBytes))
          else
            Left(CudaDriverFailure.fromStatus(s"CUDA pooled device allocation of $classBytes bytes", result.status))
        }.map { block =>
          liveDevice :+= block
          liveBytesCount += block.classBytes
          new CudaDeviceBuffer[T](
            context = context,
            valueType = valueType,
            elementCount = elementCount,
            sizeBytes = sizeBytes,
            handle = block.handle,
            backend = backend,
            releaseOverride = Some(() => returnDevice(block))
          )
        }
    }

  def allocatePinned[T](elementCount: Int)(using
      valueType: CudaType[T]
  ): Either[CudaDriverFailure, CudaPinnedBuffer[T]] =
    context.synchronizedLifecycle {
      requireOpenPool()
      require(
        elementCount > 0,
        s"pooled pinned buffer element count must be positive: $elementCount"
      )
      val sizeBytes = Math.multiplyExact(elementCount.toLong, valueType.sizeBytes.toLong)
      require(
        sizeBytes <= Int.MaxValue,
        s"CUDA pinned buffers require at most ${Int.MaxValue} bytes: $sizeBytes"
      )
      if sizeBytes > config.maxBlockBytes then
        passThroughs += 1
        context.allocatePinned[T](elementCount)
      else
        val classBytes = roundUp(sizeBytes)
        takePinned(classBytes).map(Right(_)).getOrElse {
          val result = backend.allocatePinnedMemory(context.nativeHandle, classBytes)
          if result.status.succeeded then
            require(result.handle != 0L, "successful CUDA pinned allocation returned a null address")
            require(
              result.storage != null && result.storage.isDirect &&
                result.storage.capacity() == classBytes.toInt,
              "successful CUDA pinned allocation returned invalid storage"
            )
            misses += 1
            Right(new PinnedBlock(result.handle, classBytes, result.storage))
          else
            Left(CudaDriverFailure.fromStatus(s"CUDA pooled pinned allocation of $classBytes bytes", result.status))
        }.map { block =>
          livePinned :+= block
          liveBytesCount += block.classBytes
          new CudaPinnedBuffer[T](
            context = context,
            valueType = valueType,
            elementCount = elementCount,
            sizeBytes = sizeBytes,
            handle = block.handle,
            storage = block.storage.slice(0, Math.toIntExact(sizeBytes)).order(ByteOrder.nativeOrder()),
            backend = backend,
            releaseOverride = Some(() => returnPinned(block))
          )
        }
    }

  def withBuffer[T, A](elementCount: Int)(
      body: CudaDeviceBuffer[T] => Either[CudaDriverFailure, A]
  )(using CudaType[T]): Either[CudaDriverFailure, A] =
    allocate[T](elementCount).flatMap(loan(_)(body))

  def withPinned[T, A](elementCount: Int)(
      body: CudaPinnedBuffer[T] => Either[CudaDriverFailure, A]
  )(using CudaType[T]): Either[CudaDriverFailure, A] =
    allocatePinned[T](elementCount).flatMap(loan(_)(body))

  def trim(): Either[CudaDriverFailure, Unit] = context.synchronizedLifecycle {
    requireOpenPool()
    trimLocked()
  }

  def close(): Either[CudaDriverFailure, Unit] = context.synchronizedLifecycle {
    if closed then Right(())
    else if liveDevice.nonEmpty || livePinned.nonEmpty then
      Left(
        CudaDriverFailure(
          operation = "CUDA memory pool close",
          resultCode = -1,
          resultName = "FLIGHT4S_POOL_LIVE_BUFFERS",
          resultDescription =
            s"the pool still owns ${liveDevice.size + livePinned.size} live buffers",
          infoLog = "",
          errorLog = ""
        )
      )
    else
      trimLocked().map { _ =>
        closed = true
        context.unregister(contextResource)
      }
  }

  // Failures keep the body's result precedence documented in the design note.
  private def loan[R <: AutoCloseable, A](resource: R)(
      body: R => Either[CudaDriverFailure, A]
  ): Either[CudaDriverFailure, A] =
    val result =
      try body(resource)
      catch
        case error: Throwable =>
          closeRecording(resource)
          throw error
    result match
      case left @ Left(_) =>
        closeRecording(resource)
        left
      case Right(value) =>
        try
          resource.close()
          Right(value)
        catch case error: CudaDriverException => Left(error.failure)

  private def closeRecording(resource: AutoCloseable): Unit =
    try resource.close()
    catch
      case _: CudaDriverException =>
        context.synchronizedLifecycle { failedReleases += 1 }

  private def requireOpenPool(): Unit =
    context.requireOpen()
    if closed then throw IllegalStateException("CUDA memory pool is closed")

  private def roundUp(bytes: Long): Long =
    var classBytes = 256L
    while classBytes < bytes do classBytes <<= 1
    classBytes

  private def takeDevice(classBytes: Long): Option[DeviceBlock] =
    val index = deviceCached.indexWhere(_.classBytes == classBytes)
    if index < 0 then None
    else
      val block = deviceCached(index)
      deviceCached = deviceCached.patch(index, Nil, 1)
      cachedBytesCount -= classBytes
      hits += 1
      Some(block)

  private def takePinned(classBytes: Long): Option[PinnedBlock] =
    val index = pinnedCached.indexWhere(_.classBytes == classBytes)
    if index < 0 then None
    else
      val block = pinnedCached(index)
      pinnedCached = pinnedCached.patch(index, Nil, 1)
      cachedBytesCount -= classBytes
      hits += 1
      Some(block)

  private def returnDevice(block: DeviceBlock): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      liveDevice = liveDevice.filterNot(_ eq block)
      liveBytesCount -= block.classBytes
      if closed then
        val status = backend.freeDeviceMemory(context.nativeHandle, block.handle)
        if status.succeeded then Right(())
        else Left(CudaDriverFailure.fromStatus("CUDA pooled device free", status))
      else if cachedBytesCount + block.classBytes > config.maxCachedBytes then
        val status = backend.freeDeviceMemory(context.nativeHandle, block.handle)
        if status.succeeded then Right(())
        else
          // Keep the block cached for a later trim; the close itself succeeds.
          failedReleases += 1
          deviceCached :+= block
          cachedBytesCount += block.classBytes
          Right(())
      else
        deviceCached :+= block
        cachedBytesCount += block.classBytes
        Right(())
    }

  private def returnPinned(block: PinnedBlock): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      livePinned = livePinned.filterNot(_ eq block)
      liveBytesCount -= block.classBytes
      if closed then
        val status = backend.freePinnedMemory(context.nativeHandle, block.handle)
        if status.succeeded then Right(())
        else Left(CudaDriverFailure.fromStatus("CUDA pooled pinned free", status))
      else if cachedBytesCount + block.classBytes > config.maxCachedBytes then
        val status = backend.freePinnedMemory(context.nativeHandle, block.handle)
        if status.succeeded then Right(())
        else
          failedReleases += 1
          pinnedCached :+= block
          cachedBytesCount += block.classBytes
          Right(())
      else
        pinnedCached :+= block
        cachedBytesCount += block.classBytes
        Right(())
    }

  private def trimLocked(): Either[CudaDriverFailure, Unit] =
    var firstFailure = Option.empty[CudaDriverFailure]
    deviceCached = deviceCached.filter { block =>
      val status = backend.freeDeviceMemory(context.nativeHandle, block.handle)
      if status.succeeded then
        cachedBytesCount -= block.classBytes
        false
      else
        if firstFailure.isEmpty then
          firstFailure = Some(CudaDriverFailure.fromStatus(
            s"CUDA pooled device free of ${block.classBytes} bytes", status))
        true
    }
    pinnedCached = pinnedCached.filter { block =>
      val status = backend.freePinnedMemory(context.nativeHandle, block.handle)
      if status.succeeded then
        cachedBytesCount -= block.classBytes
        false
      else
        if firstFailure.isEmpty then
          firstFailure = Some(CudaDriverFailure.fromStatus(
            s"CUDA pooled pinned free of ${block.classBytes} bytes", status))
        true
    }
    firstFailure.toLeft(())

  private def closeFromContext(): Unit = context.synchronizedLifecycle {
    if !closed then
      // The context is closing: live blocks can never be used again, free them too.
      liveDevice.foreach { block =>
        deviceCached :+= block
        cachedBytesCount += block.classBytes
      }
      livePinned.foreach { block =>
        pinnedCached :+= block
        cachedBytesCount += block.classBytes
      }
      liveDevice = Vector.empty
      livePinned = Vector.empty
      liveBytesCount = 0L
      trimLocked() match
        case Left(failure) => throw CudaDriverException(failure)
        case Right(()) => closed = true
  }
