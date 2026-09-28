package flight4s.core.compiler

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{ReductionPolicy, SourceSpan}
import flight4s.core.launch.{Block as LaunchBlock}
import flight4s.core.unsafe.raw.RawCuda

class NvrtcCompilationKeySuite extends FunSuite:
  private val target = ComputeCapability(8, 0)
  private val version = NvrtcVersion(13, 0)
  private val programName = "copy_values.cu"

  test("identical compilation inputs produce one stable SHA-256 key"):
    val generated = copyModule()

    val first = derive(generated)
    val second = derive(generated)

    assertEquals(first, second)
    assertEquals(first.hex.length, 64)
    assert(first.hex.forall { character =>
      character >= '0' && character <= '9' ||
      character >= 'a' && character <= 'f'
    })
    assertEquals(first.toString, first.hex)
    assertEquals(
      first.hex,
      "d79e1f0407656ce46b83b5d975dfaeffb5cb89fe4f0d93dbb32de1b3a1600fd4"
    )
    assertEquals(
      derive(generated, codegenVersion = 29).hex,
      "de75d7223869e1a6e40bbf30d3ace8f858c4e04ab5f0a8f82ebdf6108ff82081"
    )
    assertEquals(
      derive(generated, codegenVersion = 28).hex,
      "05c2c86c30f43a6c2a3f0b3ed197056ae1a95554fb4de383af08d06b7d210958"
    )
    assertEquals(
      derive(generated, codegenVersion = 27).hex,
      "52bb053370e2d53a108574f058cace020e292f2a520ff5f6598fc903533fa17e"
    )
    assertEquals(
      derive(generated, codegenVersion = 26).hex,
      "b9b580fc49e902fff93b5696d69420c5874d9ced747dc2691582eeedb0b7e295"
    )
    assertEquals(
      derive(generated, codegenVersion = 25).hex,
      "9e1bfbdbfe9584e6e362405632bf2bd47476b6724d74f8e22793d9ecd187d732"
    )
    assertEquals(
      derive(generated, codegenVersion = 24).hex,
      "2d1359c08ae1c08efad87c0b31b77278803af0f6bb6aafdef0197cbff97e5661"
    )
    // The previous codegen emitted uncast unsigned CUDA indexing fields.
    assertNotEquals(
      first.hex,
      "8198ba43bd8f1503bb636ed03612dd333e86796418d48467e7cce37629d78717"
    )
    // Encoding v3 must not reuse any of the earlier v2 cache identities.
    assertNotEquals(
      first.hex,
      "dfdcb6216b4a704f4c680cd88f605dcc17d33b2d2dae46ed437da3e766055b3a"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 22).hex,
      "2a5a273905247900df1ff1637b8be816805032e7cf6a748b3079aad64bcdd8e7"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 21).hex,
      "75442bd854c4b03269dd2ba7719772c18de4236e9d34e9321afc5e8c274d19f9"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 20).hex,
      "010a2abc0ebb8ee4f5c63d21fc34fa9d4f56075ffe16ac45c8173baec5c8fd48"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 19).hex,
      "a7e9688c9002e414620aa0b518bc774fa29f181d7c1dab755ab3870eda3be1a3"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 18).hex,
      "d4728173a7b8019141f5c5b6020d15b33cad86c917444b70aba90d19940c0210"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 17).hex,
      "dd9af418098caac63e6dd4d162c20f886536c875e8a0faeaf60ad00db929a74a"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 16).hex,
      "67b29366699bf4ae3afc244cfb1d49cbf41b45cc6848ccf923438a8bff09369c"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 15).hex,
      "fb1222db1773487068514c23b8ac3acf4e8b0011ea1eb80657c78b7f7e1c0b9d"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 14).hex,
      "a81e163de06c5214d927beef4d16c41719be4175695420761a565a6743702512"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 13).hex,
      "8b4310d2d5248b8e529f51f8df3f1ac2c9294be8d3b42ed9c0c5738d3a29f0cc"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 12).hex,
      "9802f54c26f291e6d1be5104989e716b4c191e2b2b2b04a45d9139aa609a1028"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 11).hex,
      "4b93dfe58695696b324af592b81ae871dc8b6430da2cc0ffb67d48b3caff34da"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 10).hex,
      "a30d5164a26f12b34d08f9ac274d661936bfedd4b380e8f8e05eea447240bd13"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 9).hex,
      "124cd591c0dd3a6123e8cd3d025f6e9d501577b622d9da75471677280389373d"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 8).hex,
      "20704f17112945b4c073dc98c55f72479ca415947600f0e121e4d76e7756e693"
    )
    assertNotEquals(
      derive(generated, codegenVersion = 7).hex,
      "e03e074906e92c6e7c00e58e46ca8b0db07916bdc014b9e2be8062c672ad5b64"
    )

  test("required block shapes affect generated and raw identities including every dimension"):
    val generated = copyModule()
    val shapes = Vector(None, Some(LaunchBlock.x(128)), Some(LaunchBlock.xy(64, 2)),
      Some(LaunchBlock.xyz(32, 2, 2)), Some(LaunchBlock.xyz(32, 4, 1)))
    val generatedKeys = shapes.map { shape =>
      derive(generated.copy(kernels = generated.kernels.map(kernel =>
        withLaunchRequirements(kernel, kernel.launchRequirements.copy(requiredBlock = shape)))))
    }
    assertEquals(generatedKeys.distinct.size, shapes.size)
    val signature = params()
    val rawKeys = shapes.map { shape =>
      val raw = RawCuda.kernel("rawShape", signature, "extern \"C\" __global__ void rawShape() {}",
        CompilerOptions(), KernelLaunchRequirements(requiredBlock = shape))
      NvrtcCompilationKey.derive(flight4s.core.compiler.NvrtcCompilationInput.raw(raw), target, version, programName)
    }
    assertEquals(rawKeys.distinct.size, shapes.size)

  test("every compiler-relevant input invalidates the key"):
    val generated = copyModule()
    val alternateMetadata = scalarModule()
    val baseKey = derive(generated)
    val changedOptions = generated.copy(
      compilerOptions = CompilerOptions(
        additionalNvrtcOptions = Vector("--use_fast_math")
      )
    )
    val changedMetadata = generated.copy(
      kernels = alternateMetadata.kernels
    )
    val changedLaunchRequirements = generated.copy(
      kernels = generated.kernels.map { kernel =>
        withLaunchRequirements(
          kernel,
          KernelLaunchRequirements(
            dynamicSharedMemory = Some(
              DynamicSharedMemoryRequirement(4, 4)
            )
          )
        )
      }
    )

    val variants = Vector(
      "CUDA source" -> derive(
        generated.copy(cudaSource = generated.cudaSource + "// changed\n")
      ),
      "compiler options" -> derive(changedOptions),
      "compute capability" -> derive(
        generated,
        target = ComputeCapability(9, 0)
      ),
      "NVRTC version" -> derive(
        generated,
        nvrtcVersion = NvrtcVersion(13, 1)
      ),
      "program name" -> derive(
        generated,
        programName = "renamed.cu"
      ),
      "codegen version" -> derive(
        generated,
        codegenVersion = CudaCodegen.ArtifactVersion + 1
      ),
      "kernel ABI metadata" -> derive(changedMetadata),
      "kernel launch requirements" -> derive(changedLaunchRequirements)
    )

    variants.foreach { case (label, variantKey) =>
      assert(baseKey != variantKey, s"$label did not invalidate the key")
    }

  test("canonical option encoding preserves element boundaries and order"):
    val generated = copyModule()
    val splitAfter = generated.copy(
      compilerOptions = CompilerOptions(
        additionalNvrtcOptions = Vector("ab", "c")
      )
    )
    val splitBefore = generated.copy(
      compilerOptions = CompilerOptions(
        additionalNvrtcOptions = Vector("a", "bc")
      )
    )
    val reordered = generated.copy(
      compilerOptions = CompilerOptions(
        additionalNvrtcOptions = Vector("c", "ab")
      )
    )

    assertNotEquals(derive(splitAfter), derive(splitBefore))
    assertNotEquals(derive(splitAfter), derive(reordered))

  test("source maps and generated line metadata do not affect PTX identity"):
    val generated = copyModule()
    val replacementMap = SourceMap(
      Vector(
        SourceMapEntry(
          generatedLine = 99,
          sourceSpan = SourceSpan("Other.scala", 80, 7, 80, 42)
        )
      )
    )
    val remapped = generated.copy(
      sourceMap = replacementMap,
      kernels = generated.kernels.map { kernel =>
        withSourceMetadata(kernel, replacementMap)
      }
    )

    assertEquals(derive(generated), derive(remapped))

  test("reduction policies produce distinct compilation identities"):
    val strict = derive(reductionModule(ReductionPolicy.Strict))
    val deterministic = derive(reductionModule(ReductionPolicy.Deterministic))
    val fast = derive(reductionModule(ReductionPolicy.Fast))

    assertNotEquals(strict, deterministic)
    assertNotEquals(strict, fast)
    assertNotEquals(deterministic, fast)

  test("raw CUDA identity includes source, ABI, launch, options, and environment"):
    val signature = params(
      input[Float]("source"),
      output[Float]("destination")
    )
    val source =
      "extern \"C\" __global__ void rawCopy(const float* source, float* destination) {}"
    val definition = RawCuda.kernel(
      "rawCopy",
      signature,
      source,
      CompilerOptions(),
      KernelLaunchRequirements()
    )
    val baseKey = deriveInput(NvrtcCompilationInput.raw(definition))
    val changedAbi = RawCuda.kernel(
      "rawCopy",
      params(value[Float]("source"), output[Float]("destination")),
      source,
      CompilerOptions(),
      KernelLaunchRequirements()
    )
    val changedLaunch = RawCuda.kernel(
      "rawCopy",
      signature,
      source,
      CompilerOptions(),
      KernelLaunchRequirements(Some(DynamicSharedMemoryRequirement(4, 4)))
    )
    val changedOptions = RawCuda.kernel(
      "rawCopy",
      signature,
      source,
      CompilerOptions(additionalNvrtcOptions = Vector("--use_fast_math")),
      KernelLaunchRequirements()
    )

    val variants = Vector(
      "source" -> deriveInput(
        NvrtcCompilationInput.raw(
          RawCuda.kernel(
            "rawCopy",
            signature,
            source + "\n// changed",
            CompilerOptions(),
            KernelLaunchRequirements()
          )
        )
      ),
      "entry point" -> deriveInput(
        NvrtcCompilationInput.raw(
          RawCuda.kernel(
            "renamedRawCopy",
            signature,
            source,
            CompilerOptions(),
            KernelLaunchRequirements()
          )
        )
      ),
      "ABI" -> deriveInput(NvrtcCompilationInput.raw(changedAbi)),
      "launch requirements" -> deriveInput(
        NvrtcCompilationInput.raw(changedLaunch)
      ),
      "compiler options" -> deriveInput(
        NvrtcCompilationInput.raw(changedOptions)
      ),
      "compute capability" -> deriveInput(
        NvrtcCompilationInput.raw(definition),
        target = ComputeCapability(9, 0)
      ),
      "NVRTC version" -> deriveInput(
        NvrtcCompilationInput.raw(definition),
        nvrtcVersion = NvrtcVersion(13, 1)
      ),
      "program name" -> deriveInput(
        NvrtcCompilationInput.raw(definition),
        programName = "renamed.cu"
      )
    )

    variants.foreach { case (label, variantKey) =>
      assertNotEquals(baseKey, variantKey, s"$label did not invalidate the raw key")
    }

  test("generated and raw provenance cannot share a compilation identity"):
    val signature = params(value[Int]("count"))
    val source = "extern \"C\" __global__ void sameSource(int count) {}"
    val options = CompilerOptions()
    val requirements = KernelLaunchRequirements()
    val generatedKernel = GeneratedKernel(
      "sameSource",
      signature,
      source,
      SourceMap(Vector.empty),
      options,
      declarationLine = 1,
      requirements
    )
    val generated = GeneratedCudaModule(
      source,
      SourceMap(Vector.empty),
      options,
      Vector(generatedKernel)
    )
    val raw = RawCuda.kernel(
      "sameSource",
      signature,
      source,
      options,
      requirements
    )

    assertNotEquals(
      deriveInput(NvrtcCompilationInput.generated(generated)),
      deriveInput(NvrtcCompilationInput.raw(raw))
    )

  test("invalid key inputs are rejected before hashing"):
    val generated = copyModule()

    intercept[IllegalArgumentException](
      derive(generated.copy(cudaSource = ""))
    )
    intercept[IllegalArgumentException](
      derive(generated, programName = "")
    )
    intercept[IllegalArgumentException](
      derive(generated, programName = "bad\u0000name.cu")
    )
    intercept[IllegalArgumentException](
      derive(generated, codegenVersion = 0)
    )

  private def derive(
      generated: GeneratedCudaModule,
      target: ComputeCapability = target,
      nvrtcVersion: NvrtcVersion = version,
      programName: String = programName,
      codegenVersion: Int = CudaCodegen.ArtifactVersion
  ): NvrtcCompilationKey =
    NvrtcCompilationKey.derive(
      generated,
      target,
      nvrtcVersion,
      programName,
      codegenVersion
    )

  private def deriveInput(
      input: NvrtcCompilationInput,
      target: ComputeCapability = target,
      nvrtcVersion: NvrtcVersion = version,
      programName: String = programName
  ): NvrtcCompilationKey =
    NvrtcCompilationKey.derive(
      input,
      target,
      nvrtcVersion,
      programName
    )

  private def copyModule(
      compilerOptions: CompilerOptions = CompilerOptions()
  ): GeneratedCudaModule =
    val source = input[Float]("source")
    val destination = output[Float]("destination")
    val definition = kernel(
      "copyValues",
      params(source, destination)
    ) { bindings =>
      bindings.tail.head(threadIdx.x) :=
        bindings.head(threadIdx.x).read
    }

    CudaCodegen
      .generateModule(
        module(kernels = Vector(definition)),
        compilerOptions
      )
      .toOption
      .get

  private def scalarModule(): GeneratedCudaModule =
    val scalar = value[Float]("scalar")
    val destination = output[Float]("destination")
    val definition = kernel(
      "fillValues",
      params(scalar, destination)
    ) { bindings =>
      bindings.tail.head(threadIdx.x) := bindings.head
    }

    CudaCodegen
      .generateModule(module(kernels = Vector(definition)))
      .toOption
      .get

  private def reductionModule(
      policy: ReductionPolicy
  ): GeneratedCudaModule =
    val destination = output[Float]("destination")
    val definition = kernel(
      "reduceValues",
      params(destination)
    ) { bindings =>
      val sum = reduceSum(
        "i",
        literal(0),
        literal(4),
        literal(0.0f),
        policy
      )(_ => literal(1.0f))
      bindings.head(literal(0)) := sum
    }

    CudaCodegen
      .generateModule(module(kernels = Vector(definition)))
      .toOption
      .get

  private def withLaunchRequirements[Args <: Tuple](
      kernel: GeneratedKernel[Args],
      requirements: KernelLaunchRequirements
  ): GeneratedKernel[Args] =
    kernel.copy(launchRequirements = requirements)

  private def withSourceMetadata[Args <: Tuple](
      kernel: GeneratedKernel[Args],
      sourceMap: SourceMap
  ): GeneratedKernel[Args] =
    kernel.copy(
      sourceMap = sourceMap,
      declarationLine = kernel.declarationLine + 100
    )
