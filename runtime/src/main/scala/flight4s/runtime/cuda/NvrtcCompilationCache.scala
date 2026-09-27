package flight4s.runtime.cuda

import java.util.LinkedHashMap
import java.util.concurrent.{CompletableFuture, ConcurrentHashMap, ExecutionException}

import flight4s.core.codegen.GeneratedCudaModule
import flight4s.core.compiler.*
import flight4s.core.unsafe.raw.RawCudaKernel

/**
 * A caller-owned, bounded cache of successful NVRTC PTX compilations.
 *
 * Cached payloads deliberately exclude caller-specific compilation-input
 * metadata. Each result is rebound to the input supplied by the current caller.
 */
final class NvrtcCompilationCache private[cuda] (
    val maximumEntries: Int,
    private val backend: NvrtcCompilerBackend,
    private val artifactStore: Option[NvrtcArtifactStore] = None
):
  require(maximumEntries > 0, "cache maximumEntries must be positive")

  private val cacheLock = Object()
  private val completed =
    LinkedHashMap[NvrtcCompilationKey, CachedArtifact](16, 0.75f, true)
  private val inFlight =
    ConcurrentHashMap[
      NvrtcCompilationKey,
      CompletableFuture[CachedCompilation]
    ]()

  /** The number of completed successful entries currently retained. */
  def entryCount: Int =
    cacheLock.synchronized(completed.size())

  /** Removes completed entries without cancelling an NVRTC compilation in progress. */
  def clear(): Unit =
    cacheLock.synchronized(completed.clear())

  /**
   * Removes completed persistent entries without changing the in-memory cache.
   *
   * A cache without an artifact store treats this as a no-op.
   */
  def clearPersistent(): Either[NvrtcArtifactStoreError, Unit] =
    artifactStore.fold[Either[NvrtcArtifactStoreError, Unit]](Right(()))(
      _.clear()
    )

  def compile(
      input: NvrtcCompilationInput,
      target: ComputeCapability
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    compile(input, target, NvrtcCompiler.DefaultProgramName)

  def compile(
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    NvrtcCompiler.validateRequest(input, programName)

    backend.version() match
      case Left(failure) => Left(failure)
      case Right(version) =>
        val key = NvrtcCompilationKey.derive(
          input,
          target,
          version,
          programName
        )

        completedArtifact(key) match
          case Some(artifact) => artifact.rebind(input)
          case None => compileOrAwait(key, input, target, programName)

  def compile(
      generated: GeneratedCudaModule,
      target: ComputeCapability
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    compile(generated, target, NvrtcCompiler.DefaultProgramName)

  def compile(
      generated: GeneratedCudaModule,
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    compile(
      NvrtcCompilationInput.generated(generated),
      target,
      programName
    )

  def compile[Args <: Tuple](
      raw: RawCudaKernel[Args],
      target: ComputeCapability
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    compile(raw, target, NvrtcCompiler.DefaultProgramName)

  def compile[Args <: Tuple](
      raw: RawCudaKernel[Args],
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    compile(NvrtcCompilationInput.raw(raw), target, programName)

  private def compileOrAwait(
      key: NvrtcCompilationKey,
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    val created = CompletableFuture[CachedCompilation]()
    val existing = inFlight.putIfAbsent(key, created)

    if existing == null then
      compileOwned(key, input, target, programName, created)
    else await(existing).rebind(input)

  private def compileOwned(
      key: NvrtcCompilationKey,
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      programName: String,
      future: CompletableFuture[CachedCompilation]
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    try
      completedArtifact(key) match
        case Some(artifact) =>
          future.complete(artifact)
          artifact.rebind(input)
        case None =>
          persistentArtifact(key, input) match
            case Some(artifact) =>
              store(key, artifact)
              future.complete(artifact)
              artifact.rebind(input)
            case None =>
              backend.compile(input, target, programName) match
                case Right(artifact) =>
                  val cached = CachedArtifact.from(artifact)
                  store(key, cached)
                  persist(key, artifact)
                  future.complete(cached)
                  cached.rebind(input)
                case Left(failure) =>
                  val cached = CachedFailure.from(failure)
                  future.complete(cached)
                  cached.rebind(input)
    catch
      case exception: Throwable =>
        future.completeExceptionally(exception)
        throw exception
    finally inFlight.remove(key, future)

  private def completedArtifact(
      key: NvrtcCompilationKey
  ): Option[CachedArtifact] =
    cacheLock.synchronized(Option(completed.get(key)))

  private def store(
      key: NvrtcCompilationKey,
      artifact: CachedArtifact
  ): Unit =
    cacheLock.synchronized {
      completed.put(key, artifact)
      while completed.size() > maximumEntries do
        val iterator = completed.entrySet().iterator()
        iterator.next()
        iterator.remove()
    }

  private def persistentArtifact(
      key: NvrtcCompilationKey,
      input: NvrtcCompilationInput
  ): Option[CachedArtifact] =
    input match
      case generated: NvrtcCompilationInput.Generated =>
        artifactStore.flatMap { store =>
          store.load(key, generated) match
            case Right(Some(artifact)) => Some(CachedArtifact.from(artifact))
            case Right(None) => None
            case Left(_: NvrtcArtifactStoreIoFailure) => None
            case Left(_) =>
              store.remove(key)
              None
        }
      case _: NvrtcCompilationInput.Raw[?] => None

  private def persist(
      key: NvrtcCompilationKey,
      artifact: NvrtcArtifact
  ): Unit =
    artifact.input match
      case _: NvrtcCompilationInput.Generated =>
        artifactStore.foreach(_.store(key, artifact))
      case _: NvrtcCompilationInput.Raw[?] => ()

  private def await(
      future: CompletableFuture[CachedCompilation]
  ): CachedCompilation =
    try future.get()
    catch
      case _: InterruptedException =>
        Thread.currentThread().interrupt()
        throw IllegalStateException("interrupted while waiting for NVRTC compilation")
      case exception: ExecutionException =>
        exception.getCause match
          case runtime: RuntimeException => throw runtime
          case error: Error => throw error
          case cause =>
            throw IllegalStateException(
              "NVRTC compilation failed unexpectedly",
              cause
            )

private[cuda] object NvrtcCompilationCache:
  def apply(maximumEntries: Int): NvrtcCompilationCache =
    NvrtcCompilationCache(maximumEntries, NativeNvrtcCompilerBackend)

  def persistent(
      maximumEntries: Int,
      artifactStore: NvrtcArtifactStore
  ): NvrtcCompilationCache =
    new NvrtcCompilationCache(
      maximumEntries,
      NativeNvrtcCompilerBackend,
      Some(artifactStore)
    )

  private[cuda] def apply(
      maximumEntries: Int,
      backend: NvrtcCompilerBackend
  ): NvrtcCompilationCache =
    new NvrtcCompilationCache(maximumEntries, backend)

  private[cuda] def persistent(
      maximumEntries: Int,
      backend: NvrtcCompilerBackend,
      artifactStore: NvrtcArtifactStore
  ): NvrtcCompilationCache =
    new NvrtcCompilationCache(maximumEntries, backend, Some(artifactStore))

private sealed trait CachedCompilation:
  def rebind(
      input: NvrtcCompilationInput
  ): Either[NvrtcCompilationError, NvrtcArtifact]

private final case class CachedArtifact(
    ptx: Array[Byte],
    compileLog: String,
    nvrtcVersion: NvrtcVersion,
    target: ComputeCapability,
    compilerOptions: NvrtcCompileOptions,
    programName: String
) extends CachedCompilation:
  override def rebind(
      input: NvrtcCompilationInput
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    Right(
      NvrtcArtifact(
        input = input,
        ptx = IArray.unsafeFromArray(ptx.clone()),
        compileLog = compileLog,
        nvrtcVersion = nvrtcVersion,
        target = target,
        compilerOptions = compilerOptions,
        programName = programName
      )
    )

private object CachedArtifact:
  def from(artifact: NvrtcArtifact): CachedArtifact =
    CachedArtifact(
      ptx = IArray.genericWrapArray(artifact.ptx).toArray,
      compileLog = artifact.compileLog,
      nvrtcVersion = artifact.nvrtcVersion,
      target = artifact.target,
      compilerOptions = artifact.compilerOptions,
      programName = artifact.programName
    )

private final case class CachedFailure(
    resultCode: Int,
    resultName: String,
    compileLog: String,
    nvrtcVersion: NvrtcVersion,
    target: ComputeCapability,
    compilerOptions: NvrtcCompileOptions,
    programName: String
) extends CachedCompilation:
  override def rebind(
      input: NvrtcCompilationInput
  ): Either[NvrtcCompilationError, NvrtcArtifact] =
    Left(
      NvrtcCompileFailure(
        input = input,
        resultCode = resultCode,
        resultName = resultName,
        compileLog = compileLog,
        nvrtcVersion = nvrtcVersion,
        target = target,
        compilerOptions = compilerOptions,
        programName = programName
      )
    )

private object CachedFailure:
  def from(failure: NvrtcCompileFailure): CachedFailure =
    CachedFailure(
      resultCode = failure.resultCode,
      resultName = failure.resultName,
      compileLog = failure.compileLog,
      nvrtcVersion = failure.nvrtcVersion,
      target = failure.target,
      compilerOptions = failure.compilerOptions,
      programName = failure.programName
    )
