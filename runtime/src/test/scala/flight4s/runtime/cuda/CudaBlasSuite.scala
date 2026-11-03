package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.runtime.cuda.internal.NativeCudaDriverStatus

class CudaBlasSuite extends FunSuite:
  private def openedContext(backend: RecordingBackend): CudaContext =
    CudaContext.open(0, backend).toOption.get

  private def gemmEvents(backend: RecordingBackend): Int =
    backend.events.count(_.startsWith("gemm:"))
  private def deviceFrees(backend: RecordingBackend): Int =
    backend.events.count(_.startsWith("free:"))

  private val failedStatus = NativeCudaDriverStatus(
    resultCode = 1,
    resultName = "CUBLAS_STATUS_INTERNAL_ERROR",
    resultDescription = "blas operation failed"
  )

  test("blas handles create gemm and close through recorded backend calls"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val blas = context.createBlas().toOption.get
      assertEquals(backend.events.count(_.startsWith("createBlas:")), 1)
      val a = context.allocate[Float](6).toOption.get
      val b = context.allocate[Float](12).toOption.get
      val c = context.allocate[Float](8).toOption.get
      assertEquals(blas.gemmF32(a, b, c, GemmShape(m = 2, n = 4, k = 3)), Right(()))
      assertEquals(gemmEvents(backend), 1)
      assert(backend.events.exists(_.startsWith("gemm:")), backend.events.mkString("\n"))
      val stream = context.createStream().toOption.get
      assertEquals(blas.gemmF32(a, b, c, GemmShape(2, 4, 3), stream = Some(stream)), Right(()))
      assertEquals(gemmEvents(backend), 2)
      val workspace = context.allocate[Float](256).toOption.get
      assertEquals(blas.gemmF32(a, b, c, GemmShape(2, 4, 3), alpha = 0.5f, beta = 1.0f,
        workspace = Some(workspace)), Right(()))
      assertEquals(gemmEvents(backend), 3)
      assert(backend.events.exists(event => event.startsWith("gemm:") && event.endsWith(":1024")),
        backend.events.mkString("\n"))
      assertEquals(context.synchronize(), Right(()))
      assertEquals(stream.synchronize(), Right(()))
      assertEquals(blas.close(), Right(()))
      assertEquals(backend.events.count(_.startsWith("destroyBlas:")), 1)
      assertEquals(blas.close(), Right(()))
      assertEquals(backend.events.count(_.startsWith("destroyBlas:")), 1)
    finally context.close()

  test("shapes and element counts validate before any native call"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val blas = context.createBlas().toOption.get
      val a = context.allocate[Float](6).toOption.get
      val b = context.allocate[Float](12).toOption.get
      val c = context.allocate[Float](8).toOption.get
      intercept[IllegalArgumentException](GemmShape(0, 1, 1))
      intercept[IllegalArgumentException](blas.gemmF32(a, b, c, GemmShape(3, 4, 3)))
      intercept[IllegalArgumentException](blas.gemmF32(a, b, c, GemmShape(2, 4, 4)))
      intercept[IllegalArgumentException](blas.gemmF32(a, b, a, GemmShape(2, 4, 3)))
      assertEquals(gemmEvents(backend), 0)
      assertEquals(blas.close(), Right(()))
      intercept[IllegalStateException](blas.gemmF32(a, b, c, GemmShape(2, 4, 3)))
      assertEquals(gemmEvents(backend), 0)
    finally context.close()

  test("failures propagate from create execute and destroy with retry"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      backend.createBlasStatus = failedStatus
      assert(context.createBlas().isLeft)
      backend.createBlasStatus = RecordingBackend.successStatus
      val blas = context.createBlas().toOption.get
      val a = context.allocate[Float](1).toOption.get
      val b = context.allocate[Float](1).toOption.get
      val c = context.allocate[Float](1).toOption.get
      backend.gemmStatus = failedStatus
      assert(blas.gemmF32(a, b, c, GemmShape(1, 1, 1)).isLeft)
      // A failed submission releases its leases at once: this close frees immediately.
      c.close()
      assertEquals(deviceFrees(backend), 1)
      backend.gemmStatus = RecordingBackend.successStatus
      backend.destroyBlasStatus = failedStatus
      assert(blas.close().isLeft)
      backend.destroyBlasStatus = RecordingBackend.successStatus
      assertEquals(blas.close(), Right(()))
    finally context.close()

  test("gemm defers buffer closes until completion"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    try
      val blas = context.createBlas().toOption.get
      val a = context.allocate[Float](4).toOption.get
      val b = context.allocate[Float](4).toOption.get
      val c = context.allocate[Float](4).toOption.get
      assertEquals(blas.gemmF32(a, b, c, GemmShape(2, 2, 2)), Right(()))
      c.close()
      assertEquals(deviceFrees(backend), 0)
      assertEquals(context.synchronize(), Right(()))
      assertEquals(deviceFrees(backend), 1)
      val stream = context.createStream().toOption.get
      assertEquals(blas.gemmF32(a, b, a, GemmShape(2, 2, 2), stream = Some(stream)), Right(()))
      b.close()
      assertEquals(deviceFrees(backend), 1)
      assertEquals(stream.synchronize(), Right(()))
      assertEquals(deviceFrees(backend), 2)
      assertEquals(blas.close(), Right(()))
    finally context.close()

  test("closing the context destroys an open blas handle"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    val blas = context.createBlas().toOption.get
    assertEquals(backend.events.count(_.startsWith("createBlas:")), 1)
    context.close()
    assertEquals(backend.events.count(_.startsWith("destroyBlas:")), 1)
