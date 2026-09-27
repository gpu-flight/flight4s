package flight4s.runtime.cuda

import java.nio.charset.StandardCharsets

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.compiler.*
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.SourceSpan
import flight4s.core.unsafe.raw.RawCuda

class NvrtcCompilerSuite extends FunSuite:
  private val nativeLibraryConfigured =
    sys.props.contains("flight4s.cuda.native.path")

  test("NVRTC version query matches generated CUDA compilation"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val queriedVersion = NvrtcCompiler.version() match
      case Right(value) => value
      case Left(failure) => fail(failure.message)
    assert(queriedVersion.major >= 12)
    assert(queriedVersion.minor >= 0)

    val inputBuffer = input[Float]("input")
    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "copyValues",
      params(inputBuffer, outputBuffer)
    ) { bindings =>
      val input = bindings.head
      val output = bindings.tail.head
      output(threadIdx.x) := input(threadIdx.x).read
    }
    val generated = CudaCodegen.generate(definition) match
      case Right(value) =>
        GeneratedCudaModule(
          value.cudaSource,
          value.sourceMap,
          value.compilerOptions,
          Vector(value)
        )
      case Left(error) =>
        fail(error.message)

    NvrtcCompiler.compile(
      generated,
      ComputeCapability(8, 0),
      "copy_values.cu"
    ) match
      case Right(artifact) =>
        val ptxText = String(
          IArray.genericWrapArray(artifact.ptx).toArray,
          StandardCharsets.UTF_8
        )

        assert(artifact.ptx.nonEmpty)
        assert(ptxText.contains(".entry copyValues"))
        assertEquals(artifact.nvrtcVersion, queriedVersion)
        assertEquals(artifact.target, ComputeCapability(8, 0))
        assertEquals(artifact.programName, "copy_values.cu")
        assertEquals(
          artifact.compilerOptions.values,
          Vector(
            "--std=c++20",
            "--gpu-architecture=compute_80"
          )
        )
        assertGeneratedInput(artifact.input, generated)
      case Left(failure) =>
        fail(failure.message + "\n" + failure.compileLog)

  test("NVRTC compilation failures retain source, options, and logs"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val brokenSpan = SourceSpan(
      file = "BrokenKernel.scala",
      startLine = 12,
      startColumn = 5,
      endLine = 12,
      endColumn = 41
    )
    val generated = GeneratedCudaModule(
      cudaSource =
        """extern "C" __global__ void broken( {
          |}
          |""".stripMargin.replace("\r\n", "\n"),
      sourceMap = SourceMap(
        Vector(SourceMapEntry(1, brokenSpan))
      ),
      compilerOptions = CompilerOptions(),
      kernels = Vector.empty
    )

    NvrtcCompiler.compile(
      generated,
      ComputeCapability(8, 0),
      "broken_kernel.cu"
    ) match
      case Left(failure) =>
        assertEquals(
          failure.resultName,
          "NVRTC_ERROR_COMPILATION"
        )
        assert(failure.resultCode != 0)
        assert(failure.compileLog.nonEmpty)
        assert(failure.compileLog.contains("broken_kernel.cu"))
        val diagnostic = failure.diagnostics
          .find(_.severity == NvrtcDiagnosticSeverity.Error)
          .getOrElse(fail("NVRTC error diagnostic was not parsed"))
        assertEquals(
          diagnostic.generatedLocation.file,
          "broken_kernel.cu"
        )
        assertEquals(diagnostic.generatedLocation.line, 1)
        assertEquals(diagnostic.sourceSpan, Some(brokenSpan))
        assert(
          diagnostic.render.contains(
            "Scala source: BrokenKernel.scala:12:5"
          )
        )
        assertGeneratedInput(failure.input, generated)
        assertEquals(
          failure.compilerOptions.values,
          Vector(
            "--std=c++20",
            "--gpu-architecture=compute_80"
          )
        )
      case Right(_) =>
        fail("invalid CUDA source unexpectedly compiled")

  test("NVRTC compiles caller-provided raw CUDA without changing its source"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val definition = RawCuda.kernel(
      entryPoint = "rawSaturate",
      signature = params(output[Float]("output")),
      source =
        """extern "C" __global__ void rawSaturate(float* output) {
          |  output[threadIdx.x] = __saturatef(1.5f);
          |}
          |""".stripMargin,
      compilerOptions = CompilerOptions(),
      launchRequirements = KernelLaunchRequirements()
    )
    NvrtcCompiler.compile(
      definition,
      ComputeCapability(8, 0),
      "raw_saturate.cu"
    ) match
      case Right(artifact) =>
        val ptxText = String(
          IArray.genericWrapArray(artifact.ptx).toArray,
          StandardCharsets.UTF_8
        )

        artifact.input match
          case input: NvrtcCompilationInput.Raw[?] =>
            assert(input.definition eq definition)
          case _: NvrtcCompilationInput.Generated =>
            fail("expected caller-provided raw source provenance")
        assertEquals(
          artifact.provenance,
          NvrtcSourceProvenance.CallerProvidedRaw
        )
        assertEquals(artifact.input.source, definition.source)
        assert(ptxText.contains(".entry rawSaturate"))
      case Left(failure) =>
        fail(failure.message + "\n" + failure.compileLog)

  test("NVRTC diagnostics remap generated statements to Scala call sites"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "capturedBrokenStatement",
      params(outputBuffer)
    ) { bindings =>
      bindings.head(threadIdx.x) := literal(1.0f)
    }
    val generatedKernel = CudaCodegen.generate(definition) match
      case Right(value) => value
      case Left(error) => fail(error.message)
    val brokenSource = generatedKernel.cudaSource.replace(
      "0x1.0p0f",
      "missingGeneratedSymbol"
    )
    assertNotEquals(brokenSource, generatedKernel.cudaSource)
    val generated = GeneratedCudaModule(
      cudaSource = brokenSource,
      sourceMap = generatedKernel.sourceMap,
      compilerOptions = generatedKernel.compilerOptions,
      kernels = Vector(generatedKernel)
    )
    val expectedSpan = generated.sourceMap.entries.last.sourceSpan

    NvrtcCompiler.compile(
      generated,
      ComputeCapability(8, 0),
      "captured_broken_statement.cu"
    ) match
      case Left(failure) =>
        val diagnostic = failure.diagnostics
          .find(_.severity == NvrtcDiagnosticSeverity.Error)
          .getOrElse(fail("NVRTC error diagnostic was not parsed"))
        assertEquals(diagnostic.sourceSpan, Some(expectedSpan))
        assert(
          expectedSpan.file
            .replace('\\', '/')
            .endsWith("NvrtcCompilerSuite.scala")
        )
        assert(expectedSpan.startLine > 0)
        assert(expectedSpan.startColumn > 0)
        assert(
          diagnostic.render.contains(
            s"Scala source: ${expectedSpan.file}:" +
              s"${expectedSpan.startLine}:${expectedSpan.startColumn}"
          )
        )
      case Right(_) =>
        fail("corrupted generated CUDA unexpectedly compiled")

  test("compile request validates source and logical program name before JNI"):
    val empty = GeneratedCudaModule(
      cudaSource = "",
      sourceMap = SourceMap(Vector.empty),
      compilerOptions = CompilerOptions(),
      kernels = Vector.empty
    )
    val nonEmpty = empty.copy(cudaSource = " ")
    val target = ComputeCapability(8, 0)

    intercept[IllegalArgumentException](
      NvrtcCompiler.compile(empty, target)
    )
    intercept[IllegalArgumentException](
      NvrtcCompiler.compile(nonEmpty, target, "")
    )
    intercept[IllegalArgumentException](
      NvrtcCompiler.compile(nonEmpty, target, "bad\u0000name.cu")
    )

  private def assertGeneratedInput(
      input: NvrtcCompilationInput,
      expected: GeneratedCudaModule
  ): Unit =
    input match
      case generated: NvrtcCompilationInput.Generated =>
        assertEquals(generated.module, expected)
      case _: NvrtcCompilationInput.Raw[?] =>
        fail("expected a DSL-generated NVRTC compilation input")
