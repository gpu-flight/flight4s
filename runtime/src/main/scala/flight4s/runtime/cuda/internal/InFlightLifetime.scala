package flight4s.runtime.cuda.internal

private[flight4s] final class InFlightResourceState[E] private (
    releaseNative: () => Either[E, Unit]
):
  private var closeRequested = false
  private var released = false
  private var leaseCount = 0
  private var retainCount = 0

  def isOpen: Boolean = !closeRequested && !released
  def isReleased: Boolean = released
  // Retained leases pin lifetime without counting as pending activity.
  def isInFlight: Boolean = leaseCount > 0

  def requireIdle(description: String): Unit =
    if isInFlight then
      throw IllegalStateException(s"$description has in-flight work")

  def acquire(): InFlightLease[E] =
    if !isOpen then
      throw IllegalStateException("cannot retain a closed CUDA resource")
    leaseCount += 1
    new InFlightLease(this, retained = false)

  def acquireRetained(): InFlightLease[E] =
    if !isOpen then
      throw IllegalStateException("cannot retain a closed CUDA resource")
    retainCount += 1
    new InFlightLease(this, retained = true)

  def requestClose(): Either[E, Unit] =
    if released then Right(())
    else
      closeRequested = true
      releaseIfReady()

  private[internal] def releaseLease(retained: Boolean): Either[E, Unit] =
    if retained then
      require(retainCount > 0, "retained lease count underflow")
      retainCount -= 1
    else
      require(leaseCount > 0, "in-flight lease count underflow")
      leaseCount -= 1
    releaseIfReady()

  private def releaseIfReady(): Either[E, Unit] =
    if closeRequested && leaseCount == 0 && retainCount == 0 && !released then
      releaseNative() match
        case Right(()) =>
          released = true
          Right(())
        case Left(failure) =>
          closeRequested = false
          Left(failure)
    else Right(())

private[flight4s] object InFlightResourceState:
  def apply[E](
      releaseNative: () => Either[E, Unit]
  ): InFlightResourceState[E] =
    new InFlightResourceState(releaseNative)

private[flight4s] final class InFlightLease[E] private[internal] (
    owner: InFlightResourceState[E],
    retained: Boolean = false
):
  private var released = false

  def release(): Either[E, Unit] =
    if released then Right(())
    else
      released = true
      owner.releaseLease(retained)

private[flight4s] final class InFlightBatch[E](
    leases: Vector[InFlightLease[E]]
):
  private var completed = false

  def isComplete: Boolean = completed

  def complete(): Either[E, Unit] =
    if completed then Right(())
    else
      var firstFailure = Option.empty[E]
      leases.foreach { lease =>
        lease.release() match
          case Left(failure) if firstFailure.isEmpty =>
            firstFailure = Some(failure)
          case _ => ()
      }
      completed = true
      firstFailure.toLeft(())

private[flight4s] final class InFlightTracker[E]:
  private var pendingBatches = Vector.empty[InFlightBatch[E]]

  def hasPending: Boolean =
    pruneCompleted()
    pendingBatches.nonEmpty

  def add(batch: InFlightBatch[E]): Unit =
    pruneCompleted()
    pendingBatches :+= batch

  def snapshot(): Vector[InFlightBatch[E]] =
    pruneCompleted()
    pendingBatches

  def completePending(): Either[E, Unit] =
    complete(snapshot())

  def complete(
      batches: Vector[InFlightBatch[E]]
  ): Either[E, Unit] =
    var firstFailure = Option.empty[E]
    batches.foreach { batch =>
      batch.complete() match
        case Left(failure) if firstFailure.isEmpty =>
          firstFailure = Some(failure)
        case _ => ()
    }
    pruneCompleted()
    firstFailure.toLeft(())

  private def pruneCompleted(): Unit =
    pendingBatches = pendingBatches.filterNot(_.isComplete)
