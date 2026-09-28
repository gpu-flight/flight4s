package flight4s.runtime.cuda

import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.attribute.BasicFileAttributes
import java.nio.file.{FileVisitResult, Files, Path, SimpleFileVisitor}
import java.util.concurrent.{CountDownLatch, Executors, Future, TimeUnit}
import java.util.concurrent.atomic.AtomicReference

import munit.FunSuite

import flight4s.core.codegen.{
  CompilerOptions,
  GeneratedCudaModule,
  KernelLaunchRequirements,
  SourceMap,
  SourceMapEntry
}
import flight4s.core.compiler.*
import flight4s.core.dsl.CudaDsl.params
import flight4s.core.ir.SourceSpan
import flight4s.core.unsafe.raw.RawCuda

class NvrtcCompilationCacheSuite extends FunSuite:
  private val target = ComputeCapability(8, 0)
  private val version = NvrtcVersion(13, 0)
  private val programName = "cached_kernel.cu"

  test("a cache hit rebinds PTX metadata to the current generated module"):
    val first = generated("FirstCallSite.scala")
    val remapped = first.copy(sourceMap = sourceMap("SecondCallSite.scala"))
    val backend = RecordingBackend()
    val cache = NvrtcCompilationCache(2, backend)

    val initialArtifact = compiled(cache, first)
    val cachedArtifact = compiled(cache, remapped)

    assertEquals(backend.compileCount, 1)
    assertEquals(cache.entryCount, 1)
    assertEquals(generatedModule(initialArtifact.input), first)
    assertEquals(generatedModule(cachedArtifact.input), remapped)
    assertEquals(cachedArtifact.input.sourceMap, remapped.sourceMap)
    assertEquals(ptxText(cachedArtifact), ptxText(initialArtifact))

  test("compiler-relevant input changes miss the cache"):
    val backend = RecordingBackend()
    val cache = NvrtcCompilationCache(3, backend)
    val initial = generated("Initial.scala")

    compiled(cache, initial)
    compiled(
      cache,
      initial.copy(cudaSource = initial.cudaSource + "// changed\n")
    )
    compiled(cache, initial, target = ComputeCapability(9, 0))

    assertEquals(backend.compileCount, 3)
    assertEquals(cache.entryCount, 3)

  test("a raw cache hit rebinds PTX metadata to the current input"):
    val first = rawInput()
    val rebound = rawInput()
    val backend = RecordingBackend()
    val cache = NvrtcCompilationCache(2, backend)

    val initialArtifact = compiled(cache, first)
    val cachedArtifact = compiled(cache, rebound)

    assertEquals(backend.compileCount, 1)
    assertEquals(cache.entryCount, 1)
    assert(initialArtifact.input eq first)
    assert(cachedArtifact.input eq rebound)
    assertEquals(
      cachedArtifact.provenance,
      NvrtcSourceProvenance.CallerProvidedRaw
    )
    assertEquals(ptxText(cachedArtifact), ptxText(initialArtifact))

  test("failed compilations are shared in flight but not retained"):
    val backend = RecordingBackend()
    backend.compileAction = (generated, target, programName, attempt) =>
      if attempt == 1 then
        Left(failedCompilation(generated, target, programName))
      else Right(successfulArtifact(generated, target, programName))
    val cache = NvrtcCompilationCache(2, backend)
    val module = generated("Retry.scala")

    val first = cache.compile(module, target, programName)
    val second = cache.compile(module, target, programName)

    assert(first.isLeft)
    assert(second.isRight)
    assertEquals(backend.compileCount, 2)
    assertEquals(cache.entryCount, 1)

  test("least recently used completed entries are evicted at the configured bound"):
    val backend = RecordingBackend()
    val cache = NvrtcCompilationCache(2, backend)
    val first = generated("First.scala")
    val second = first.copy(cudaSource = first.cudaSource + "// second\n")
    val third = first.copy(cudaSource = first.cudaSource + "// third\n")

    compiled(cache, first)
    compiled(cache, second)
    compiled(cache, first)
    compiled(cache, third)
    compiled(cache, second)

    assertEquals(backend.compileCount, 4)
    assertEquals(cache.entryCount, 2)

  test("concurrent matching requests compile once and receive independent artifacts"):
    val backend = RecordingBackend()
    val started = CountDownLatch(1)
    val release = CountDownLatch(1)
    backend.pauseCompile(started, release)
    val cache = NvrtcCompilationCache(2, backend)
    val module = generated("Concurrent.scala")
    val executor = Executors.newFixedThreadPool(2)

    try
      val first = executor.submit(() => cache.compile(module, target, programName))
      assert(started.await(1, TimeUnit.SECONDS), "first compilation did not start")

      val waiter = AtomicReference[Thread]()
      val second = executor.submit(() => {
        waiter.set(Thread.currentThread())
        cache.compile(module, target, programName)
      })
      awaitWaitingRequest(waiter, second, backend)

      release.countDown()
      val firstArtifact = first.get(1, TimeUnit.SECONDS) match
        case Right(value) => value
        case Left(error) => fail(error.message)
      val secondArtifact = second.get(1, TimeUnit.SECONDS) match
        case Right(value) => value
        case Left(error) => fail(error.message)

      assertEquals(backend.compileCount, 1)
      assertEquals(generatedModule(firstArtifact.input), module)
      assertEquals(generatedModule(secondArtifact.input), module)
      assert(!(firstArtifact.ptx.asInstanceOf[AnyRef] eq secondArtifact.ptx.asInstanceOf[AnyRef]))
    finally
      release.countDown()
      executor.shutdownNow()

  test("concurrent failed requests share one result and later requests retry"):
    val backend = RecordingBackend()
    val started = CountDownLatch(1)
    val release = CountDownLatch(1)
    backend.pauseCompile(started, release)
    backend.compileAction = (generated, target, programName, _) =>
      Left(failedCompilation(generated, target, programName))
    val cache = NvrtcCompilationCache(2, backend)
    val module = generated("ConcurrentFailure.scala")
    val executor = Executors.newFixedThreadPool(2)

    try
      val first = executor.submit(() => cache.compile(module, target, programName))
      assert(started.await(1, TimeUnit.SECONDS), "first compilation did not start")
      val waiter = AtomicReference[Thread]()
      val second = executor.submit(() => {
        waiter.set(Thread.currentThread())
        cache.compile(module, target, programName)
      })
      awaitWaitingRequest(waiter, second, backend)

      release.countDown()
      assert(first.get(1, TimeUnit.SECONDS).isLeft)
      assert(second.get(1, TimeUnit.SECONDS).isLeft)
      assertEquals(backend.compileCount, 1)
      assertEquals(cache.entryCount, 0)

      backend.compileAction = (generated, target, programName, _) =>
        Right(successfulArtifact(generated, target, programName))
      assert(compiled(cache, module).ptx.nonEmpty)
      assertEquals(backend.compileCount, 2)
      assertEquals(cache.entryCount, 1)
    finally
      release.countDown()
      executor.shutdownNow()

  test("version-query failures do not invoke compilation or create entries"):
    val versionFailure = NvrtcVersionQueryFailure(
      resultCode = 7,
      resultName = "NVRTC_ERROR_INTERNAL_ERROR"
    )
    val backend = RecordingBackend(Left(versionFailure))
    val cache = NvrtcCompilationCache(2, backend)

    val result = cache.compile(generated("Version.scala"), target, programName)

    assertEquals(result, Left(versionFailure))
    assertEquals(backend.compileCount, 0)
    assertEquals(cache.entryCount, 0)

  test("persistent cache reloads disk after a memory clear and clears each layer explicitly"):
    withStore { store =>
      val backend = RecordingBackend()
      val cache = NvrtcCompilationCache.persistent(2, backend, store)
      val initial = generated("PersistentInitial.scala")
      val remapped = initial.copy(sourceMap = sourceMap("PersistentRemapped.scala"))

      compiled(cache, initial)
      cache.clear()
      val fromDisk = compiled(cache, remapped)

      assertEquals(backend.compileCount, 1)
      assertEquals(generatedModule(fromDisk.input), remapped)
      assertEquals(cache.entryCount, 1)

      assertEquals(cache.clearPersistent(), Right(()))
      assertEquals(cache.entryCount, 1)
      cache.clear()
      compiled(cache, remapped)

      assertEquals(backend.compileCount, 2)
    }

  test("persistent cache reloads raw artifacts after memory is cleared"):
    withStore { store =>
      val backend = RecordingBackend()
      val cache = NvrtcCompilationCache.persistent(2, backend, store)

      val original = compiled(cache, rawInput())
      cache.clear()
      val rebound = rawInput()
      val loaded = compiled(cache, rebound)

      assertEquals(backend.compileCount, 1)
      assertEquals(cache.entryCount, 1)
      assert(loaded.input eq rebound)
      assertEquals(ptxText(loaded), ptxText(original))
      assertEquals(loaded.diagnostics.head.sourceSpan, None)

      val restartedBackend = RecordingBackend()
      val restarted = NvrtcCompilationCache.persistent(
        2, restartedBackend, NvrtcArtifactStore(store.directory)
      )
      val fromDisk = compiled(restarted, rawInput())
      assertEquals(restartedBackend.compileCount, 0)
      assertEquals(ptxText(fromDisk), ptxText(original))
      assert(!(fromDisk.ptx.asInstanceOf[AnyRef] eq original.ptx.asInstanceOf[AnyRef]))

      assertEquals(restarted.clearPersistent(), Right(()))
      restarted.clear()
      compiled(restarted, rawInput())
      assertEquals(restartedBackend.compileCount, 1)
    }

  test("persistent raw entries recover from corrupt payloads and provenance"):
    Vector("artifact.ptx", "source.cu", "compile.log", "manifest").foreach { file =>
      withStore { store =>
        val backend = RecordingBackend()
        val cache = NvrtcCompilationCache.persistent(2, backend, store)
        val input = rawInput()
        val key = NvrtcCompilationKey.derive(input, target, version, programName)
        compiled(cache, input)
        cache.clear()
        val path = store.entryPath(key).resolve(file)
        val corrupt = if file == "manifest" then
          Files.readString(path).replace("source.provenance=raw", "source.provenance=unknown")
        else "corrupt payload"
        Files.writeString(path, corrupt)

        compiled(cache, input)
        assertEquals(backend.compileCount, 2)
        cache.clear()
        compiled(cache, rawInput())
        assertEquals(backend.compileCount, 2)
      }
    }

  test("persistent cache replaces schema 1 entries after recompilation"):
    withStore { store =>
      val backend = RecordingBackend()
      val cache = NvrtcCompilationCache.persistent(2, backend, store)
      val module = generated("Legacy.scala")
      val key = NvrtcCompilationKey.derive(module, target, version, programName)
      compiled(cache, module)
      cache.clear()
      val entry = store.entryPath(key)
      val manifest = entry.resolve("manifest")
      Files.writeString(manifest, Files.readString(manifest).linesIterator
        .filterNot(_.startsWith("source."))
        .map(_.replace("schema=2", "schema=1")).mkString("", "\n", "\n"))
      Files.move(entry.resolve("source.cu"), entry.resolve("generated.cu"))

      compiled(cache, module)
      assertEquals(backend.compileCount, 2)
      assert(!Files.exists(entry.resolve("generated.cu")))
      assert(Files.readString(manifest).contains("schema=2\n"))
      cache.clear()
      compiled(cache, module)
      assertEquals(backend.compileCount, 2)
    }

  test("raw compilation failures never create persistent entries"):
    withStore { store =>
      val backend = RecordingBackend()
      backend.compileAction = (input, target, name, _) =>
        Left(failedCompilation(input, target, name))
      val cache = NvrtcCompilationCache.persistent(2, backend, store)
      val input = rawInput()
      val key = NvrtcCompilationKey.derive(input, target, version, programName)

      assert(cache.compile(input, target, programName).isLeft)
      assert(cache.compile(input, target, programName).isLeft)
      assertEquals(backend.compileCount, 2)
      assertEquals(cache.entryCount, 0)
      assert(!Files.exists(store.entryPath(key)))
    }

  test("persistent generated inputs retain their explicit codegen version"):
    withStore { store =>
      val backend = RecordingBackend()
      val cache = NvrtcCompilationCache.persistent(2, backend, store)
      val initialModule = generated("VersionedInitial.scala")
      val remappedModule = initialModule.copy(
        sourceMap = sourceMap("VersionedRemapped.scala")
      )
      val initial = NvrtcCompilationInput.generated(
        initialModule,
        codegenVersion = 19
      )
      val remapped = NvrtcCompilationInput.generated(
        remappedModule,
        codegenVersion = 19
      )

      compiled(cache, initial)
      cache.clear()
      val fromDisk = compiled(cache, remapped)

      assertEquals(backend.compileCount, 1)
      assert(fromDisk.input eq remapped)
    }

  test("persistent cache removes a corrupt entry, recompiles, and repairs the store"):
    withStore { store =>
      val backend = RecordingBackend()
      val cache = NvrtcCompilationCache.persistent(2, backend, store)
      val module = generated("PersistentRepair.scala")
      val key = NvrtcCompilationKey.derive(module, target, version, programName)

      compiled(cache, module)
      cache.clear()
      Files.writeString(store.entryPath(key).resolve("artifact.ptx"), "corrupted")

      compiled(cache, module)
      assertEquals(backend.compileCount, 2)

      cache.clear()
      compiled(cache, module)
      assertEquals(backend.compileCount, 2)
    }

  test("persistent cache falls back to NVRTC when artifact-store I/O fails"):
    val rootFile = Files.createTempFile("flight4s-nvrtc-cache-", ".tmp")
    try
      val backend = RecordingBackend()
      val cache = NvrtcCompilationCache.persistent(
        2,
        backend,
        NvrtcArtifactStore(rootFile)
      )

      compiled(cache, generated("StoreFailure.scala"))

      assertEquals(backend.compileCount, 1)
      assertEquals(cache.entryCount, 1)
    finally Files.deleteIfExists(rootFile)

  test("persistent cache still compiles one matching request at a time"):
    withStore { store =>
      val backend = RecordingBackend()
      val started = CountDownLatch(1)
      val release = CountDownLatch(1)
      backend.pauseCompile(started, release)
      val cache = NvrtcCompilationCache.persistent(2, backend, store)
      val module = generated("PersistentConcurrent.scala")
      val executor = Executors.newFixedThreadPool(2)

      try
        val first = executor.submit(() => cache.compile(module, target, programName))
        assert(started.await(1, TimeUnit.SECONDS), "first compilation did not start")
        val waiter = AtomicReference[Thread]()
        val second = executor.submit(() => {
          waiter.set(Thread.currentThread())
          cache.compile(module, target, programName)
        })
        awaitWaitingRequest(waiter, second, backend)

        release.countDown()
        assert(first.get(1, TimeUnit.SECONDS).isRight)
        assert(second.get(1, TimeUnit.SECONDS).isRight)
        assertEquals(backend.compileCount, 1)
      finally
        release.countDown()
        executor.shutdownNow()
    }

  private def compiled(
      cache: NvrtcCompilationCache,
      generated: GeneratedCudaModule,
      target: ComputeCapability = target
  ): NvrtcArtifact =
    cache.compile(generated, target, programName) match
      case Right(artifact) => artifact
      case Left(error) => fail(error.message)

  private def compiled(
      cache: NvrtcCompilationCache,
      input: NvrtcCompilationInput
  ): NvrtcArtifact =
    cache.compile(input, target, programName) match
      case Right(artifact) => artifact
      case Left(error) => fail(error.message)

  private def generated(sourceFile: String): GeneratedCudaModule =
    GeneratedCudaModule(
      cudaSource =
        "extern \"C\" __global__ void cachedKernel() {}\n",
      sourceMap = sourceMap(sourceFile),
      compilerOptions = CompilerOptions(),
      kernels = Vector.empty
    )

  private def sourceMap(sourceFile: String): SourceMap =
    SourceMap(
      Vector(
        SourceMapEntry(
          generatedLine = 1,
          sourceSpan = SourceSpan(sourceFile, 10, 3, 10, 27)
        )
      )
    )

  private def rawInput(): NvrtcCompilationInput.Raw[EmptyTuple] =
    NvrtcCompilationInput.raw(
      RawCuda.kernel(
        entryPoint = "rawCachedKernel",
        signature = params(),
        source =
          "extern \"C\" __global__ void rawCachedKernel() {}\n",
        compilerOptions = CompilerOptions(),
        launchRequirements = KernelLaunchRequirements()
      )
    )

  private def successfulArtifact(
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      programName: String
  ): NvrtcArtifact =
    NvrtcArtifact(
      input = input,
      ptx = IArray.unsafeFromArray(
        ".version 8.0\n.entry cachedKernel() {}\n"
          .getBytes(StandardCharsets.UTF_8)
      ),
      compileLog = "cached_kernel.cu(1): warning: cached compilation",
      nvrtcVersion = version,
      target = target,
      compilerOptions = NvrtcCompileOptions.resolve(
        input.compilerOptions,
        target
      ),
      programName = programName
    )

  private def failedCompilation(
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      programName: String
  ): NvrtcCompileFailure =
    NvrtcCompileFailure(
      input = input,
      resultCode = 6,
      resultName = "NVRTC_ERROR_COMPILATION",
      compileLog = "cached_kernel.cu(1): error: broken source",
      nvrtcVersion = version,
      target = target,
      compilerOptions = NvrtcCompileOptions.resolve(
        input.compilerOptions,
        target
      ),
      programName = programName
    )

  private def ptxText(artifact: NvrtcArtifact): String =
    String(
      IArray.genericWrapArray(artifact.ptx).toArray,
      StandardCharsets.UTF_8
    )

  private def generatedModule(
      input: NvrtcCompilationInput
  ): GeneratedCudaModule =
    input match
      case generated: NvrtcCompilationInput.Generated => generated.module
      case _: NvrtcCompilationInput.Raw[?] =>
        fail("expected a DSL-generated NVRTC compilation input")

  private def awaitWaitingRequest(
      waiter: AtomicReference[Thread],
      request: Future[?],
      backend: RecordingBackend
  ): Unit =
    // A version query precedes key derivation and does not prove in-flight registration.
    // The owner is held on a latch; a second backend call would also park, so count it.
    awaitCondition(Option(waiter.get()).exists(_.getState == Thread.State.WAITING))
    assert(!request.isDone, "second request did not wait for the active compilation")
    assertEquals(backend.versionCount, 2)
    assertEquals(backend.compileCount, 1)

  private def awaitCondition(condition: => Boolean): Unit =
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(1)
    while !condition && System.nanoTime() < deadline do Thread.sleep(5)
    assert(condition, "condition did not become true before the timeout")

  private def withStore(test: NvrtcArtifactStore => Unit): Unit =
    val root = Files.createTempDirectory("flight4s-nvrtc-compilation-cache-")
    try test(NvrtcArtifactStore(root))
    finally deleteRecursively(root)

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      Files.walkFileTree(
        path,
        new SimpleFileVisitor[Path]:
          override def visitFile(
              file: Path,
              attributes: BasicFileAttributes
          ): FileVisitResult =
            Files.delete(file)
            FileVisitResult.CONTINUE

          override def postVisitDirectory(
              directory: Path,
              exception: IOException
          ): FileVisitResult =
            if exception != null then throw exception
            Files.delete(directory)
            FileVisitResult.CONTINUE
      )

  private final class RecordingBackend(
      private var versionResponse: Either[NvrtcVersionQueryFailure, NvrtcVersion] =
        Right(version)
  ) extends NvrtcCompilerBackend:
    private var compileAttempts = 0
    private var versionAttempts = 0
    private var compileGate = Option.empty[(CountDownLatch, CountDownLatch)]

    var compileAction:
        (NvrtcCompilationInput, ComputeCapability, String, Int) => Either[NvrtcCompileFailure, NvrtcArtifact] =
      (input, target, programName, _) =>
        Right(successfulArtifact(input, target, programName))

    def compileCount: Int = synchronized(compileAttempts)

    def versionCount: Int = synchronized(versionAttempts)

    def pauseCompile(
        started: CountDownLatch,
        release: CountDownLatch
    ): Unit = synchronized {
      compileGate = Some(started -> release)
    }

    override def version(): Either[NvrtcVersionQueryFailure, NvrtcVersion] =
      synchronized {
        versionAttempts += 1
        versionResponse
      }

    override def compile(
        input: NvrtcCompilationInput,
        target: ComputeCapability,
        programName: String
    ): Either[NvrtcCompileFailure, NvrtcArtifact] =
      val (attempt, action, gate) = synchronized {
        compileAttempts += 1
        (compileAttempts, compileAction, compileGate)
      }
      gate.foreach { case (started, release) =>
        started.countDown()
        release.await()
      }
      action(input, target, programName, attempt)
