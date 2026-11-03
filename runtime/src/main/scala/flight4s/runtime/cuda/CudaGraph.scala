package flight4s.runtime.cuda

import flight4s.runtime.cuda.internal.{CudaDriverBackend, InFlightLease, InFlightResourceState}

/** An instantiated stream-capture graph. The graph holds one lease per
  * distinct resource referenced during capture until close, which is what
  * keeps device addresses (including pooled blocks) stable across replays. */
final class CudaGraph private[cuda] (
    context: CudaContext,
    backend: CudaDriverBackend,
    graphHandle: Long,
    execHandle: Long,
    heldLeases: Vector[InFlightLease[CudaDriverFailure]]
) extends CudaInFlightResource:
  private var execDestroyed = false
  private var graphDestroyed = false

  private val lifetime = InFlightResourceState[CudaDriverFailure] { () =>
    destroyNative() match
      case Left(failure) => Left(failure)
      case Right(()) =>
        var firstFailure = Option.empty[CudaDriverFailure]
        heldLeases.foreach { lease =>
          lease.release() match
            case Left(failure) if firstFailure.isEmpty => firstFailure = Some(failure)
            case _ => ()
        }
        firstFailure.toLeft(())
  }

  // Destruction is retryable: already-destroyed parts are skipped.
  private def destroyNative(): Either[CudaDriverFailure, Unit] =
    val execResult =
      if execDestroyed then Right(())
      else
        val status = backend.destroyGraphExec(context.nativeHandle, execHandle)
        if status.succeeded then
          execDestroyed = true
          Right(())
        else Left(CudaDriverFailure.fromStatus("CUDA graph exec destruction", status))
    execResult match
      case Left(failure) => Left(failure)
      case Right(()) =>
        if graphDestroyed then Right(())
        else
          val status = backend.destroyGraph(context.nativeHandle, graphHandle)
          if status.succeeded then
            graphDestroyed = true
            Right(())
          else Left(CudaDriverFailure.fromStatus("CUDA graph destruction", status))

  def isOpen: Boolean =
    context.synchronizedLifecycle(lifetime.isOpen && context.isOpen)

  def launch(stream: CudaStream): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      requireOpen()
      val status = stream.submitTracked(Vector(this)) { streamHandle =>
        backend.launchGraph(context.nativeHandle, execHandle, streamHandle)
      }
      if status.succeeded then Right(())
      else Left(CudaDriverFailure.fromStatus("CUDA graph launch", status))
    }

  def close(): Either[CudaDriverFailure, Unit] =
    context.synchronizedLifecycle {
      if lifetime.isReleased then Right(())
      else
        context.requireOpen()
        lifetime.requestClose()
    }

  private[cuda] val contextResource: AutoCloseable = () => closeFromContext()

  private def closeFromContext(): Unit = context.synchronizedLifecycle {
    if !lifetime.isReleased then
      lifetime.requestClose() match
        case Left(failure) => throw CudaDriverException(failure)
        case Right(()) => ()
  }

  override private[cuda] def acquireInFlight(): InFlightLease[CudaDriverFailure] =
    requireOpen()
    lifetime.acquire()

  override private[cuda] def acquireRetained(): InFlightLease[CudaDriverFailure] =
    requireOpen()
    lifetime.acquireRetained()

  private[cuda] def requireOpen(): Unit =
    context.requireOpen()
    if !lifetime.isOpen then
      throw IllegalStateException("CUDA graph is closed")
