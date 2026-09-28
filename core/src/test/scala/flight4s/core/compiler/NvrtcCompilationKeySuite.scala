package flight4s.core.compiler

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{ReductionPolicy, SourceSpan}
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
      "124cd591c0dd3a6123e8cd3d025f6e9d501577b622d9da75471677280389373d"
    )
    assertEquals(
      derive(generated, codegenVersion = 8).hex,
      "20704f17112945b4c073dc98c55f72479ca415947600f0e121e4d76e7756e693"
    )
    assertEquals(
      derive(generated, codegenVersion = 7).hex,
      "e03e074906e92c6e7c00e58e46ca8b0db07916bdc014b9e2be8062c672ad5b64"
    )

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
