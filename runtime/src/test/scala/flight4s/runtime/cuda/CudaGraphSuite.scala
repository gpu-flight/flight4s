package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.runtime.cuda.internal.NativeCudaDriverStatus

class CudaGraphSuite extends FunSuite:
  private def openedContext(backend: RecordingBackend): CudaContext =
    CudaContext.open(0, backend).toOption.get

  private def count(backend: RecordingBackend, prefix: String): Int =
    backend.events.count(_.startsWith(prefix))

  private val failedStatus = NativeCudaDriverStatus(
    resultCode = 1,
    resultName = "CUDA_ERROR_UNKNOWN",
    resultDescription = "driver operation failed"
  )

  private final class Fixture(val backend: RecordingBackend):
    val context = openedContext(backend)
    val stream = context.createStream().toOption.get
    val blas = context.createBlas().toOption.get
    def buffers() =
      (context.allocate[Float](4).toOption.get,
        context.allocate[Float](4).toOption.get,
        context.allocate[Float](4).toOption.get)
    def gemm(a: CudaDeviceBuffer[Float], b: CudaDeviceBuffer[Float], c: CudaDeviceBuffer[Float]) =
      blas.gemmF32(a, b, c, GemmShape(2, 2, 2), stream = Some(stream))

  test("capture records submissions without batches and holds leases until graph close"):
    val backend = RecordingBackend()
    val fixture = Fixture(backend)
    import fixture.*
    try
      val (a, b, c) = buffers()
      val graph = stream.capture {
        assertEquals(gemm(a, b, c), Right(()))
        assertEquals(gemm(a, c, b), Right(()))
      }.toOption.get
      assertEquals(count(backend, "beginCapture:"), 1)
      assertEquals(count(backend, "endCapture:"), 1)
      assertEquals(count(backend, "instantiateGraph:"), 1)
      assertEquals(count(backend, "gemm:"), 2)
      // Captured work owns no stream batches: synchronize completes nothing.
      assertEquals(stream.synchronize(), Right(()))
      // Retained leases pin lifetime, not activity: input refreshes stay legal.
      assertEquals(a.copyFrom(Array.fill(4)(1.0f)), Right(()))
      c.close()
      assertEquals(count(backend, "free:"), 0)
      assertEquals(graph.close(), Right(()))
      assertEquals(count(backend, "destroyGraphExec:"), 1)
      assertEquals(count(backend, "destroyGraph:"), 1)
      assertEquals(count(backend, "free:"), 1)
      assertEquals(graph.close(), Right(()))
      assertEquals(count(backend, "destroyGraphExec:"), 1)
    finally context.close()

  test("replays create batches completed by synchronize and defer graph close"):
    val backend = RecordingBackend()
    val fixture = Fixture(backend)
    import fixture.*
    try
      val (a, b, c) = buffers()
      val graph = stream.capture { assertEquals(gemm(a, b, c), Right(())) }.toOption.get
      assertEquals(graph.launch(stream), Right(()))
      assertEquals(count(backend, "launchGraph:"), 1)
      assertEquals(graph.close(), Right(()))
      assertEquals(count(backend, "destroyGraphExec:"), 0)
      assertEquals(stream.synchronize(), Right(()))
      assertEquals(count(backend, "destroyGraphExec:"), 1)
      assertEquals(count(backend, "destroyGraph:"), 1)
    finally context.close()

  test("failure injection at each stage releases exactly what it acquired"):
    val backend = RecordingBackend()
    val fixture = Fixture(backend)
    import fixture.*
    try
      val (a, b, c) = buffers()
      backend.beginCaptureStatus = failedStatus
      assert(stream.capture { () }.isLeft)
      backend.beginCaptureStatus = RecordingBackend.successStatus

      backend.endCaptureStatus = failedStatus
      assert(stream.capture { assertEquals(gemm(a, b, c), Right(())) }.isLeft)
      backend.endCaptureStatus = RecordingBackend.successStatus
      c.close()
      assertEquals(count(backend, "free:"), 1)

      backend.instantiateGraphStatus = failedStatus
      assert(stream.capture { assertEquals(gemm(a, b, a), Right(())) }.isLeft)
      assertEquals(count(backend, "destroyGraph:"), 1)
      backend.instantiateGraphStatus = RecordingBackend.successStatus
      b.close()
      assertEquals(count(backend, "free:"), 2)

      val graph = stream.capture { assertEquals(gemm(a, a, a), Right(())) }.toOption.get
      backend.launchGraphStatus = failedStatus
      assert(graph.launch(stream).isLeft)
      backend.launchGraphStatus = RecordingBackend.successStatus
      backend.destroyGraphExecStatus = failedStatus
      assert(graph.close().isLeft)
      backend.destroyGraphExecStatus = RecordingBackend.successStatus
      assertEquals(graph.close(), Right(()))
      a.close()
      assertEquals(count(backend, "free:"), 3)
    finally context.close()

  test("pooled buffers in a capture stay out of the free list until graph close"):
    val backend = RecordingBackend()
    val fixture = Fixture(backend)
    import fixture.*
    try
      val pool = context.createPool(MemoryPoolConfig()).toOption.get
      val a = pool.allocate[Float](4).toOption.get
      val b = pool.allocate[Float](4).toOption.get
      val c = pool.allocate[Float](4).toOption.get
      val graph = stream.capture { assertEquals(gemm(a, b, c), Right(())) }.toOption.get
      a.close(); b.close(); c.close()
      assertEquals(pool.statistics.cachedBytes, 0L)
      val miss = pool.allocate[Float](4).toOption.get
      assertEquals(pool.statistics.misses, 4L)
      assertEquals(graph.close(), Right(()))
      assertEquals(pool.statistics.cachedBytes, 3L * 256L)
      val hit = pool.allocate[Float](4).toOption.get
      assertEquals(pool.statistics.hits, 1L)
      miss.close(); hit.close()
      assertEquals(pool.close(), Right(()))
    finally context.close()

  test("capture body exceptions abort the capture cleanly"):
    val backend = RecordingBackend()
    val fixture = Fixture(backend)
    import fixture.*
    try
      val (a, b, c) = buffers()
      intercept[IllegalStateException] {
        stream.capture {
          assertEquals(gemm(a, b, c), Right(()))
          throw IllegalStateException("body")
        }
      }
      assertEquals(count(backend, "endCapture:"), 1)
      assertEquals(count(backend, "destroyGraph:"), 1)
      c.close()
      assertEquals(count(backend, "free:"), 1)
      val graph = stream.capture { assertEquals(gemm(a, b, a), Right(())) }.toOption.get
      assertEquals(graph.close(), Right(()))
    finally context.close()

  test("closing the context destroys a live graph"):
    val backend = RecordingBackend()
    val fixture = Fixture(backend)
    import fixture.*
    val (a, b, c) = buffers()
    val graph = stream.capture { assertEquals(gemm(a, b, c), Right(())) }.toOption.get
    assert(graph.isOpen)
    context.close()
    assertEquals(count(backend, "destroyGraphExec:"), 1)
    assertEquals(count(backend, "destroyGraph:"), 1)
