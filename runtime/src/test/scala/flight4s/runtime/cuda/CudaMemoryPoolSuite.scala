package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.runtime.cuda.internal.NativeCudaDriverStatus

class CudaMemoryPoolSuite extends FunSuite:
  private def openedContext(backend: RecordingBackend): CudaContext =
    CudaContext.open(0, backend).toOption.get

  private def deviceAllocations(backend: RecordingBackend): Int =
    backend.events.count(_.startsWith("allocate:"))
  private def deviceFrees(backend: RecordingBackend): Int =
    backend.events.count(_.startsWith("free:"))
  private def pinnedAllocations(backend: RecordingBackend): Int =
    backend.events.count(_.startsWith("allocatePinned:"))
  private def pinnedFrees(backend: RecordingBackend): Int =
    backend.events.count(_.startsWith("freePinned:"))

  private val failedFree = NativeCudaDriverStatus(
    resultCode = 1,
    resultName = "CUDA_ERROR_UNKNOWN",
    resultDescription = "driver operation failed"
  )

  test("device blocks are cached per size class and reused without driver calls"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val pool = context.createPool(MemoryPoolConfig()).toOption.get
      val first = pool.allocate[Float](100).toOption.get
      assertEquals(deviceAllocations(backend), 1)
      first.close()
      assertEquals(deviceFrees(backend), 0)
      assertEquals(pool.statistics.cachedBytes, 512L)
      val second = pool.allocate[Float](120).toOption.get
      assertEquals(deviceAllocations(backend), 1)
      assertEquals(pool.statistics.hits, 1L)
      assertEquals(pool.statistics.misses, 1L)
      assertEquals(pool.statistics.liveBytes, 512L)
      val third = pool.allocate[Float](100).toOption.get
      assertEquals(deviceAllocations(backend), 2)
      assertEquals(pool.statistics.misses, 2L)
      second.close()
      third.close()
      assertEquals(pool.statistics.cachedBytes, 1024L)
      assertEquals(pool.trim(), Right(()))
      assertEquals(deviceFrees(backend), 2)
      assertEquals(pool.statistics.cachedBytes, 0L)
      assertEquals(pool.close(), Right(()))
    finally context.close()

  test("size classes stay isolated and oversized requests pass through"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val pool = context.createPool(MemoryPoolConfig(maxBlockBytes = 1024L)).toOption.get
      val small = pool.allocate[Float](100).toOption.get
      small.close()
      val medium = pool.allocate[Float](200).toOption.get
      assertEquals(deviceAllocations(backend), 2)
      val oversized = pool.allocate[Float](1000).toOption.get
      assertEquals(deviceAllocations(backend), 3)
      assertEquals(pool.statistics.passThroughs, 1L)
      oversized.close()
      assertEquals(deviceFrees(backend), 1)
      medium.close()
      assertEquals(pool.statistics.cachedBytes, 512L + 1024L)
      assertEquals(pool.close(), Right(()))
    finally context.close()

  test("in-flight pooled buffers return to the cache only after the last lease releases"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val pool = context.createPool(MemoryPoolConfig()).toOption.get
      val buffer = pool.allocate[Float](64).toOption.get
      val lease = buffer.acquireInFlight()
      buffer.close()
      assertEquals(pool.statistics.cachedBytes, 0L)
      val miss = pool.allocate[Float](64).toOption.get
      assertEquals(deviceAllocations(backend), 2)
      assertEquals(lease.release(), Right(()))
      assertEquals(pool.statistics.cachedBytes, 256L)
      val hit = pool.allocate[Float](64).toOption.get
      assertEquals(deviceAllocations(backend), 2)
      assertEquals(pool.statistics.hits, 1L)
      miss.close()
      hit.close()
      assertEquals(pool.close(), Right(()))
    finally context.close()

  test("the cache cap evicts to the driver and failed evictions stay cached for retry"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val pool = context.createPool(MemoryPoolConfig(maxCachedBytes = 512L)).toOption.get
      val first = pool.allocate[Float](100).toOption.get
      val second = pool.allocate[Float](100).toOption.get
      first.close()
      assertEquals(pool.statistics.cachedBytes, 512L)
      assertEquals(deviceFrees(backend), 0)
      second.close()
      assertEquals(deviceFrees(backend), 1)
      assertEquals(pool.statistics.cachedBytes, 512L)
      val third = pool.allocate[Float](100).toOption.get
      val fourth = pool.allocate[Float](100).toOption.get
      third.close()
      backend.freeStatus = failedFree
      fourth.close()
      assertEquals(pool.statistics.cachedBytes, 1024L)
      assertEquals(pool.statistics.failedReleases, 1L)
      assert(pool.trim().isLeft)
      backend.freeStatus = RecordingBackend.successStatus
      assertEquals(pool.trim(), Right(()))
      assertEquals(pool.statistics.cachedBytes, 0L)
      assertEquals(pool.close(), Right(()))
    finally context.close()

  test("loan helpers close on every path with documented failure precedence"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val pool = context.createPool(MemoryPoolConfig()).toOption.get
      assertEquals(pool.withBuffer[Float, Int](64)(_ => Right(42)), Right(42))
      assertEquals(pool.statistics.cachedBytes, 256L)
      val bodyFailure = CudaDriverFailure.fromStatus("body", failedFree)
      assertEquals(pool.withBuffer[Float, Int](64)(_ => Left(bodyFailure)), Left(bodyFailure))
      assertEquals(pool.statistics.cachedBytes, 256L)
      intercept[IllegalStateException] {
        pool.withBuffer[Float, Int](64)(_ => throw IllegalStateException("body"))
      }
      assertEquals(pool.statistics.cachedBytes, 256L)
      backend.allocateStatus = failedFree
      var ran = false
      assert(pool.withBuffer[Float, Int](4096)(_ => { ran = true; Right(1) }).isLeft)
      assert(!ran)
      backend.allocateStatus = RecordingBackend.successStatus
      assertEquals(pool.withPinned[Float, Int](64)(_ => Right(7)), Right(7))
      assertEquals(pinnedAllocations(backend), 1)
      assertEquals(pool.withPinned[Float, Int](64)(_ => Right(8)), Right(8))
      assertEquals(pinnedAllocations(backend), 1)
      assertEquals(pool.close(), Right(()))
      assertEquals(pinnedFrees(backend), 1)
    finally context.close()

  test("scoped pools trim and close on exit and reject live buffers at close"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val result = CudaMemoryPool.scoped(context, MemoryPoolConfig()) { pool =>
        pool.withBuffer[Float, Int](64)(_ => Right(9))
      }
      assertEquals(result, Right(9))
      assertEquals(deviceFrees(backend), 1)
      val pool = context.createPool(MemoryPoolConfig()).toOption.get
      val live = pool.allocate[Float](64).toOption.get
      assert(pool.close().isLeft)
      live.close()
      assertEquals(pool.close(), Right(()))
      assertEquals(pool.close(), Right(()))
    finally context.close()

  test("closing the context releases cached pool blocks"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    val pool = context.createPool(MemoryPoolConfig()).toOption.get
    val buffer = pool.allocate[Float](64).toOption.get
    buffer.close()
    assertEquals(deviceFrees(backend), 0)
    context.close()
    assertEquals(deviceFrees(backend), 1)
