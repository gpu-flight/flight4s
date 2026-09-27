package flight4s.runtime.cuda

import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.attribute.BasicFileAttributes
import java.nio.file.{FileVisitResult, Files, Path, SimpleFileVisitor}
import java.util.concurrent.{Executors, TimeUnit}

import scala.jdk.CollectionConverters.*

import munit.FunSuite

import flight4s.core.codegen.{
  CompilerOptions,
  DynamicSharedMemoryRequirement,
  GeneratedCudaModule,
  KernelLaunchRequirements,
  SourceMap,
  SourceMapEntry
}
import flight4s.core.compiler.*
import flight4s.core.dsl.CudaDsl.{params, value}
import flight4s.core.ir.{KernelSignature, SourceSpan}
import flight4s.core.unsafe.raw.RawCuda

class NvrtcArtifactStoreSuite extends FunSuite:
  private val target = ComputeCapability(8, 0)
  private val version = NvrtcVersion(13, 0)
  private val programName = "persistent_kernel.cu"

  test("stores a successful artifact and rebinds it to current source provenance"):
    withStore { store =>
      val initial = generated("Initial.scala")
      val remapped = initial.copy(sourceMap = sourceMap("Remapped.scala"))
      val key = compilationKey(initial)
      val initialArtifact = artifact(initial)

      assertEquals(store.store(key, initialArtifact), Right(()))

      val loaded = load(store, key, remapped)
      assertEquals(generatedModule(loaded.input), remapped)
      assertEquals(loaded.input.sourceMap, remapped.sourceMap)
      assertEquals(ptxText(loaded), ptxText(initialArtifact))
      assertEquals(loaded.compileLog, initialArtifact.compileLog)
      assertEquals(loaded.nvrtcVersion, initialArtifact.nvrtcVersion)
      assertEquals(loaded.target, initialArtifact.target)
      assertEquals(loaded.compilerOptions, initialArtifact.compilerOptions)
      assertEquals(loaded.programName, initialArtifact.programName)

      val entry = store.entryPath(key)
      assert(Files.isRegularFile(entry.resolve("manifest")))
      assert(Files.isRegularFile(entry.resolve("source.cu")))
      assert(Files.isRegularFile(entry.resolve("artifact.ptx")))
      assert(Files.isRegularFile(entry.resolve("compile.log")))
      assert(!Files.exists(entry.resolve("source-map")))
    }

  test("returns None for a missing artifact entry"):
    withStore { store =>
      val module = generated("Missing.scala")

      assertEquals(store.load(compilationKey(module), module), Right(None))
    }

  test("stores raw artifacts with explicit source provenance"):
    withStore { store =>
      val input = NvrtcCompilationInput.raw(
        RawCuda.kernel(
          entryPoint = "rawPersistentKernel",
          signature = params(),
          source =
            "extern \"C\" __global__ void rawPersistentKernel() {}\n",
          compilerOptions = CompilerOptions(),
          launchRequirements = KernelLaunchRequirements()
        )
      )
      val key = NvrtcCompilationKey.derive(
        input,
        target,
        version,
        programName
      )
      val artifact = NvrtcArtifact(
        input = input,
        ptx = IArray.unsafeFromArray(
          ".version 8.0\n.entry rawPersistentKernel() {}\n"
            .getBytes(StandardCharsets.UTF_8)
        ),
        compileLog = "",
        nvrtcVersion = version,
        target = target,
        compilerOptions = NvrtcCompileOptions.resolve(
          input.compilerOptions,
          target
        ),
        programName = programName
      )

      assertEquals(store.store(key, artifact), Right(()))
      val manifest = Files.readString(store.entryPath(key).resolve("manifest"))
      assert(manifest.contains("schema=2\n"))
      assert(manifest.contains("source.provenance=raw\n"))
      assert(!manifest.contains("source.codegen.version"))
      assertEquals(Files.readString(store.entryPath(key).resolve("source.cu")), input.source)
      val rebound = NvrtcCompilationInput.raw(
        RawCuda.kernel(
          "rawPersistentKernel", params(), input.source,
          input.compilerOptions, KernelLaunchRequirements()
        )
      )
      val loaded = store.load(key, rebound) match
        case Right(Some(result)) => result
        case other => fail(s"expected a raw artifact, found $other")
      assert(loaded.input eq rebound)
      assertEquals(loaded.provenance, NvrtcSourceProvenance.CallerProvidedRaw)
      assertEquals(ptxText(loaded), ptxText(artifact))
    }

  test("raw reload validates ABI, entry point, launch requirements, source, and options"):
    withStore { store =>
      val initial = rawInput(params(value[Int]("count")))
      val key = NvrtcCompilationKey.derive(initial, target, version, programName)
      assertEquals(store.store(key, artifact(initial)), Right(()))
      val changed = Vector(
        rawInput(params(value[Float]("count"))),
        rawInput(params(value[Int]("renamed"))),
        rawInput(params(value[Int]("count")), entryPoint = "otherKernel"),
        rawInput(params(value[Int]("count")), requirements = KernelLaunchRequirements(
          dynamicSharedMemory = Some(DynamicSharedMemoryRequirement(4, 4))
        )),
        rawInput(params(value[Int]("count")), options = CompilerOptions(
          additionalNvrtcOptions = Vector("--use_fast_math")
        ))
      )
      changed.foreach { input =>
        assert(store.load(key, input).left.exists(_.isInstanceOf[NvrtcArtifactStoreInvalidEntry]))
      }
      val changedSource = rawInput(params(value[Int]("count")), source = initial.source + "// changed\n")
      assert(store.load(key, changedSource).left.exists(_.isInstanceOf[NvrtcArtifactStoreSourceMismatch]))
    }

  test("raw diagnostics and signatures rebind to the current caller after reload"):
    withStore { store =>
      val initial = rawInput(params(value[Int]("count")))
      val rebound = rawInput(params(value[Int]("count")))
      val key = NvrtcCompilationKey.derive(initial, target, version, programName)
      assertEquals(store.store(key, artifact(initial)), Right(()))
      val loaded = store.load(key, rebound).toOption.flatten.getOrElse(fail("missing artifact"))
      assert(loaded.input eq rebound)
      assert(loaded.input.kernels.head.signature eq rebound.kernels.head.signature)
      assertEquals(loaded.diagnostics.head.generatedLocation.file, programName)
      assertEquals(loaded.diagnostics.head.sourceSpan, None)
    }

  test("manifest rejects malformed provenance and codegen version fields"):
    withStore { store =>
      val module = generated("Provenance.scala")
      val key = compilationKey(module)
      assertEquals(store.store(key, artifact(module)), Right(()))
      val path = store.entryPath(key).resolve("manifest")
      val valid = Files.readString(path)
      val codegenLine = valid.linesIterator.find(_.startsWith("source.codegen.version=")).get
      Vector(
        valid.replace("source.provenance=dsl", "source.provenance=unknown"),
        valid.replace("source.provenance=dsl\n", ""),
        valid.replace(codegenLine + "\n", ""),
        valid.replace(codegenLine, "source.codegen.version=0"),
        valid.replace(codegenLine, "source.codegen.version=not-an-integer"),
        valid.replace("source.provenance=dsl", "source.provenance=raw"),
        valid + "source.provenance=dsl\n"
      ).foreach { invalidManifest =>
        Files.writeString(path, invalidManifest)
        assert(store.load(key, module).left.exists(_.isInstanceOf[NvrtcArtifactStoreInvalidEntry]))
      }
    }

  test("manifest provenance and codegen version must match the current input"):
    withStore { store =>
      val input = rawInput(params(value[Int]("count")))
      val key = NvrtcCompilationKey.derive(input, target, version, programName)
      assertEquals(store.store(key, artifact(input)), Right(()))
      val path = store.entryPath(key).resolve("manifest")
      Files.writeString(path, Files.readString(path).replace(
        "source.provenance=raw", "source.provenance=dsl\nsource.codegen.version=19"
      ))
      assert(store.load(key, input).left.exists(_.message.contains("source provenance")))

      val module = generated("Version.scala")
      val generatedKey = compilationKey(module)
      assertEquals(store.store(generatedKey, artifact(module)), Right(()))
      val wrongVersion = NvrtcCompilationInput.generated(module, codegenVersion = 19)
      assert(store.load(generatedKey, wrongVersion).left.exists(_.message.contains("source provenance")))
    }

  test("rejects an artifact whose PTX bytes no longer match the manifest"):
    withStore { store =>
      val module = generated("CorruptPtx.scala")
      val key = compilationKey(module)
      assertEquals(store.store(key, artifact(module)), Right(()))
      Files.writeString(store.entryPath(key).resolve("artifact.ptx"), "corrupted")

      store.load(key, module) match
        case Left(error: NvrtcArtifactStoreInvalidEntry) =>
          assertEquals(error.key, key)
          assert(error.reason.contains("SHA-256 mismatch"))
        case other => fail(s"expected a checksum failure, found $other")
    }

  test("rejects an entry using an unsupported manifest schema"):
    withStore { store =>
      val module = generated("FutureSchema.scala")
      val key = compilationKey(module)
      assertEquals(store.store(key, artifact(module)), Right(()))
      val manifest = store.entryPath(key).resolve("manifest")
      Files.writeString(
        manifest,
        Files.readString(manifest).replace("schema=2", "schema=99")
      )

      store.load(key, module) match
        case Left(error: NvrtcArtifactStoreUnsupportedSchema) =>
          assertEquals(error.key, key)
          assertEquals(error.foundVersion, 99)
        case other => fail(s"expected an unsupported-schema failure, found $other")
    }

  test("schema 1 entries require recompilation"):
    withStore { store =>
      val module = generated("Legacy.scala")
      val key = compilationKey(module)
      assertEquals(store.store(key, artifact(module)), Right(()))
      val path = store.entryPath(key).resolve("manifest")
      val legacy = Files.readString(path).linesIterator
        .filterNot(_.startsWith("source."))
        .map(_.replace("schema=2", "schema=1")).mkString("", "\n", "\n")
      Files.writeString(path, legacy)
      Files.move(store.entryPath(key).resolve("source.cu"), store.entryPath(key).resolve("generated.cu"))
      store.load(key, module) match
        case Left(error: NvrtcArtifactStoreUnsupportedSchema) => assertEquals(error.foundVersion, 1)
        case other => fail(s"expected a legacy-schema failure, found $other")
    }

  test("rejects invalid compilation metadata before rebuilding an artifact"):
    withStore { store =>
      val module = generated("InvalidMetadata.scala")
      val key = compilationKey(module)
      assertEquals(store.store(key, artifact(module)), Right(()))
      val manifest = store.entryPath(key).resolve("manifest")
      Files.writeString(
        manifest,
        Files.readString(manifest).replace("nvrtc.major=13", "nvrtc.major=-1")
      )

      store.load(key, module) match
        case Left(error: NvrtcArtifactStoreInvalidEntry) =>
          assert(error.reason.contains("NVRTC version components"))
        case other => fail(s"expected an invalid-entry failure, found $other")
    }

  test("rejects metadata that does not derive the requested compilation key"):
    withStore { store =>
      val module = generated("MismatchedKey.scala")
      val key = NvrtcCompilationKey.derive(
        module,
        ComputeCapability(9, 0),
        version,
        programName
      )

      store.store(key, artifact(module)) match
        case Left(error: NvrtcArtifactStoreInvalidEntry) =>
          assert(error.reason.contains("derives key"))
        case other => fail(s"expected an invalid-entry failure, found $other")
    }

  test("rejects a persisted target and option set that no longer derives its key"):
    withStore { store =>
      val module = generated("TamperedManifest.scala")
      val key = compilationKey(module)
      assertEquals(store.store(key, artifact(module)), Right(()))
      val manifest = store.entryPath(key).resolve("manifest")
      Files.writeString(
        manifest,
        Files.readString(manifest)
          .replace("target.major=8", "target.major=9")
          .replace(
            "option.2=LS1ncHUtYXJjaGl0ZWN0dXJlPWNvbXB1dGVfODA",
            "option.2=LS1ncHUtYXJjaGl0ZWN0dXJlPWNvbXB1dGVfOTA"
          )
      )

      store.load(key, module) match
        case Left(error: NvrtcArtifactStoreInvalidEntry) =>
          assert(error.reason.contains("derives key"))
        case other => fail(s"expected an invalid-entry failure, found $other")
    }

  test("rejects an entry when its CUDA source does not match the caller"):
    withStore { store =>
      val original = generated("Original.scala")
      val changed = original.copy(cudaSource = original.cudaSource + "// changed\n")
      val key = compilationKey(original)
      assertEquals(store.store(key, artifact(original)), Right(()))

      store.load(key, changed) match
        case Left(error: NvrtcArtifactStoreSourceMismatch) =>
          assertEquals(error.key, key)
        case other => fail(s"expected a source-mismatch failure, found $other")
    }

  test("concurrent writers publish one complete entry and leave no temporary directories"):
    withStore { store =>
      val module = generated("Concurrent.scala")
      val key = compilationKey(module)
      val executor = Executors.newFixedThreadPool(2)

      try
        val first = executor.submit(() => store.store(key, artifact(module)))
        val second = executor.submit(() => store.store(key, artifact(module)))
        assertEquals(first.get(1, TimeUnit.SECONDS), Right(()))
        assertEquals(second.get(1, TimeUnit.SECONDS), Right(()))
        assertEquals(ptxText(load(store, key, module)), ptxText(artifact(module)))

        val entries = Files.list(store.entryPath(key).getParent)
        try
          val temporaryEntries = entries.iterator().asScala
            .map(_.getFileName.toString)
            .filter(_.startsWith(s".${key.toString}."))
            .toVector
          assertEquals(temporaryEntries, Vector.empty)
        finally entries.close()
      finally executor.shutdownNow()
    }

  test("remove and clear invalidate completed entries without deleting the store root"):
    withStore { store =>
      val first = generated("First.scala")
      val second = first.copy(cudaSource = first.cudaSource + "// second\n")
      val firstKey = compilationKey(first)
      val secondKey = compilationKey(second)
      assertEquals(store.store(firstKey, artifact(first)), Right(()))
      assertEquals(store.store(secondKey, artifact(second)), Right(()))

      assertEquals(store.remove(firstKey), Right(()))
      assertEquals(store.load(firstKey, first), Right(None))
      assert(load(store, secondKey, second).ptx.nonEmpty)

      assertEquals(store.clear(), Right(()))
      assertEquals(store.load(secondKey, second), Right(None))
      assert(Files.isDirectory(store.directory))
    }

  private def load(
      store: NvrtcArtifactStore,
      key: NvrtcCompilationKey,
      module: GeneratedCudaModule
  ): NvrtcArtifact =
    store.load(key, module) match
      case Right(Some(value)) => value
      case other => fail(s"expected an artifact, found $other")

  private def compilationKey(module: GeneratedCudaModule): NvrtcCompilationKey =
    NvrtcCompilationKey.derive(module, target, version, programName)

  private def generated(sourceFile: String): GeneratedCudaModule =
    GeneratedCudaModule(
      cudaSource = "extern \"C\" __global__ void persistentKernel() {}\n",
      sourceMap = sourceMap(sourceFile),
      compilerOptions = CompilerOptions(additionalNvrtcOptions = Vector("--use_fast_math")),
      kernels = Vector.empty
    )

  private def sourceMap(sourceFile: String): SourceMap =
    SourceMap(
      Vector(
        SourceMapEntry(
          generatedLine = 1,
          sourceSpan = SourceSpan(sourceFile, 8, 2, 8, 34)
        )
      )
    )

  private def artifact(module: GeneratedCudaModule): NvrtcArtifact =
    artifact(NvrtcCompilationInput.generated(module))

  private def artifact(input: NvrtcCompilationInput): NvrtcArtifact =
    NvrtcArtifact(
      input = input,
      ptx = IArray.unsafeFromArray(
        ".version 8.0\n.entry persistentKernel() {}\n"
          .getBytes(StandardCharsets.UTF_8)
      ),
      compileLog = "persistent_kernel.cu(1): warning: persisted artifact",
      nvrtcVersion = version,
      target = target,
      compilerOptions = NvrtcCompileOptions.resolve(input.compilerOptions, target),
      programName = programName
    )

  private def rawInput[Args <: Tuple](
      signature: KernelSignature[Args],
      source: String = "extern \"C\" __global__ void rawKernel(int count) {}\n",
      entryPoint: String = "rawKernel",
      requirements: KernelLaunchRequirements = KernelLaunchRequirements(),
      options: CompilerOptions = CompilerOptions()
  ): NvrtcCompilationInput.Raw[Args] =
    NvrtcCompilationInput.raw(RawCuda.kernel(entryPoint, signature, source, options, requirements))

  private def generatedModule(
      input: NvrtcCompilationInput
  ): GeneratedCudaModule =
    input match
      case generated: NvrtcCompilationInput.Generated => generated.module
      case _: NvrtcCompilationInput.Raw[?] =>
        fail("expected a DSL-generated NVRTC compilation input")

  private def ptxText(artifact: NvrtcArtifact): String =
    String(IArray.genericWrapArray(artifact.ptx).toArray, StandardCharsets.UTF_8)

  private def withStore(test: NvrtcArtifactStore => Unit): Unit =
    val root = Files.createTempDirectory("flight4s-nvrtc-artifact-store-")
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
