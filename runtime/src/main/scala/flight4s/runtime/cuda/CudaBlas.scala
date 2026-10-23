package flight4s.runtime.cuda

import flight4s.runtime.cuda.internal.CudaDriverBackend

/** Row-major GEMM dimensions: A is m x k, B is k x n, C is m x n. */
final case class GemmShape(m: Int, n: Int, k: Int):
  require(m > 0 && n > 0 && k > 0, s"GEMM dimensions must be positive: $m x $n x $k")

/** One cuBLASLt handle; GEMMs run with full FP32 compute, never TF32. */
final class CudaBlas private[cuda] (
    context: CudaContext,
    backend: CudaDriverBackend,
    handle: Long
):
  private var closed = false

  private[cuda] val contextResource: AutoCloseable = () => closeFromContext()

  def isOpen: Boolean = context.synchronizedLifecycle(!closed && context.isOpen)

  def gemmF32(
      a: CudaDeviceBuffer[Float],
      b: CudaDeviceBuffer[Float],
      c: CudaDeviceBuffer[Float],
      shape: GemmShape,
      alpha: Float = 1.0f,
      beta: Float = 0.0f,
      workspace: Option[CudaDeviceBuffer[Float]] = None,
      stream: Option[CudaStream] = None
  ): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      requireOpen()
      require(
        a.elementCount == shape.m * shape.k,
        s"A must hold ${shape.m} x ${shape.k} elements, found ${a.elementCount}"
      )
      require(
        b.elementCount == shape.k * shape.n,
        s"B must hold ${shape.k} x ${shape.n} elements, found ${b.elementCount}"
      )
      require(
        c.elementCount == shape.m * shape.n,
        s"C must hold ${shape.m} x ${shape.n} elements, found ${c.elementCount}"
      )
      val tracked: Vector[CudaInFlightResource] =
        Vector(a, b, c) ++ workspace
      def submit(streamHandle: Long) =
        backend.executeGemmF32(
          contextHandle = context.nativeHandle,
          blasHandle = handle,
          streamHandle = streamHandle,
          m = shape.m,
          n = shape.n,
          k = shape.k,
          alpha = alpha,
          beta = beta,
          aAddress = a.nativeAddress,
          bAddress = b.nativeAddress,
          cAddress = c.nativeAddress,
          workspaceAddress = workspace.map(_.nativeAddress).getOrElse(0L),
          workspaceBytes = workspace.map(_.sizeBytes).getOrElse(0L)
        )
      val status = stream match
        case Some(value) => value.submitTracked(tracked)(submit)
        case None => context.submitDefaultTracked(tracked)(submit(0L))
      if status.succeeded then Right(())
      else Left(CudaDriverFailure.fromStatus("cuBLASLt GEMM", status))
    }

  def close(): Either[CudaDriverFailure, Unit] = context.synchronizedLifecycle {
    if closed then Right(())
    else
      val status = backend.destroyBlasHandle(context.nativeHandle, handle)
      if status.succeeded then
        closed = true
        context.unregister(contextResource)
        Right(())
      else Left(CudaDriverFailure.fromStatus("cuBLASLt handle destruction", status))
  }

  private def closeFromContext(): Unit = context.synchronizedLifecycle {
    if !closed then
      val status = backend.destroyBlasHandle(context.nativeHandle, handle)
      if status.succeeded then closed = true
      else
        throw CudaDriverException(
          CudaDriverFailure.fromStatus("cuBLASLt handle destruction", status)
        )
  }

  private def requireOpen(): Unit =
    context.requireOpen()
    if closed then throw IllegalStateException("cuBLASLt handle is closed")
