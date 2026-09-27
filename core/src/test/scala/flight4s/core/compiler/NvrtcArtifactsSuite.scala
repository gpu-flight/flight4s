package flight4s.core.compiler

import munit.FunSuite

import flight4s.core.codegen.{
  CompilerOptions,
  KernelLaunchRequirements,
  SourceMap,
  SourceMapEntry
}
import flight4s.core.dsl.CudaDsl.params
import flight4s.core.ir.SourceSpan
import flight4s.core.unsafe.raw.RawCuda

class NvrtcArtifactsSuite extends FunSuite:
  test("compute capability produces the NVRTC virtual architecture"):
    val capability = ComputeCapability(8, 9)

    assertEquals(capability.major, 8)
    assertEquals(capability.minor, 9)
    assertEquals(capability.virtualArchitecture, "compute_89")
    assertEquals(
      capability.nvrtcOption,
      "--gpu-architecture=compute_89"
    )
    assertEquals(capability.toString, "8.9")

  test("compute capability dimensions are validated"):
    intercept[IllegalArgumentException](ComputeCapability(0, 0))
    intercept[IllegalArgumentException](ComputeCapability(8, -1))
    intercept[IllegalArgumentException](ComputeCapability(8, 10))

  test("resolved NVRTC options preserve order and append one target"):
    val generatedOptions = CompilerOptions(
      additionalNvrtcOptions = Vector(
        "--use_fast_math",
        "--device-debug"
      )
    )

    val resolved = NvrtcCompileOptions.resolve(
      generatedOptions,
      ComputeCapability(9, 0)
    )

    assertEquals(
      resolved.values,
      Vector(
        "--std=c++20",
        "--use_fast_math",
        "--device-debug",
        "--gpu-architecture=compute_90"
      )
    )

  test("NVRTC version rejects negative components"):
    assertEquals(NvrtcVersion(12, 8).toString, "12.8")
    intercept[IllegalArgumentException](NvrtcVersion(-1, 0))
    intercept[IllegalArgumentException](NvrtcVersion(12, -1))

  test("NVRTC version query failures retain structured result metadata"):
    val failure = NvrtcVersionQueryFailure(
      resultCode = 7,
      resultName = "NVRTC_ERROR_INTERNAL_ERROR"
    )

    assertEquals(
      failure.message,
      "NVRTC version query failed with NVRTC_ERROR_INTERNAL_ERROR (7)"
    )
    intercept[IllegalArgumentException](
      NvrtcVersionQueryFailure(0, "NVRTC_SUCCESS")
    )
    intercept[IllegalArgumentException](
      NvrtcVersionQueryFailure(7, "")
    )

  test("NVRTC diagnostics map generated lines to the closest source span"):
    val declarationSpan = SourceSpan("Kernel.scala", 10, 3, 10, 24)
    val expressionSpan = SourceSpan("Kernel.scala", 18, 7, 18, 32)
    val sourceMap = SourceMap(
      Vector(
        SourceMapEntry(3, declarationSpan),
        SourceMapEntry(9, expressionSpan)
      )
    )
    val compileLog =
      """broken_kernel.cu(4): error: expected an expression
        |broken_kernel.cu(11,7): warning #177-D: variable was declared but never referenced
        |broken_kernel.cu:2:5: note: parsing started here
        |included_header.cuh(20): error: header failure
        |1 error detected in the compilation of "broken_kernel.cu".
        |""".stripMargin

    val diagnostics = NvrtcDiagnostics.parse(
      compileLog,
      sourceMap,
      "broken_kernel.cu"
    )

    assertEquals(diagnostics.size, 4)
    assertEquals(
      diagnostics.head,
      NvrtcDiagnostic(
        severity = NvrtcDiagnosticSeverity.Error,
        code = None,
        message = "expected an expression",
        generatedLocation = NvrtcGeneratedLocation(
          file = "broken_kernel.cu",
          line = 4,
          column = None
        ),
        sourceSpan = Some(declarationSpan),
        rawLine =
          "broken_kernel.cu(4): error: expected an expression"
      )
    )
    assertEquals(
      diagnostics(1).severity,
      NvrtcDiagnosticSeverity.Warning
    )
    assertEquals(diagnostics(1).code, Some("#177-D"))
    assertEquals(diagnostics(1).generatedLocation.column, Some(7))
    assertEquals(diagnostics(1).sourceSpan, Some(expressionSpan))
    assertEquals(diagnostics(2).severity, NvrtcDiagnosticSeverity.Note)
    assertEquals(diagnostics(2).sourceSpan, None)
    assertEquals(diagnostics(3).sourceSpan, None)

  test("NVRTC diagnostic rendering preserves CUDA and Scala locations"):
    val span = SourceSpan("ReductionKernel.scala", 24, 7, 24, 38)
    val diagnostic = NvrtcDiagnostic(
      severity = NvrtcDiagnosticSeverity.Error,
      code = Some("#123"),
      message = "invalid expression",
      generatedLocation = NvrtcGeneratedLocation(
        file = "block_reduce_sum.cu",
        line = 12,
        column = Some(5)
      ),
      sourceSpan = Some(span),
      rawLine = "raw diagnostic"
    )

    assertEquals(
      diagnostic.render,
      "block_reduce_sum.cu:12:5: error #123: invalid expression\n" +
        "Scala source: ReductionKernel.scala:24:7"
    )

  test("raw compilation diagnostics preserve CUDA locations without Scala mapping"):
    val definition = RawCuda.kernel(
      entryPoint = "brokenRawKernel",
      signature = params(),
      source = "extern \"C\" __global__ void brokenRawKernel( {}",
      compilerOptions = CompilerOptions(),
      launchRequirements = KernelLaunchRequirements()
    )
    val input = NvrtcCompilationInput.raw(definition)
    val target = ComputeCapability(8, 0)
    val failure = NvrtcCompileFailure(
      input = input,
      resultCode = 6,
      resultName = "NVRTC_ERROR_COMPILATION",
      compileLog =
        "raw_kernel.cu(1,36): error: expected a closing parenthesis",
      nvrtcVersion = NvrtcVersion(13, 0),
      target = target,
      compilerOptions = NvrtcCompileOptions.resolve(
        definition.compilerOptions,
        target
      ),
      programName = "raw_kernel.cu"
    )

    val diagnostic = failure.diagnostics.head
    assert(failure.input eq input)
    assertEquals(failure.provenance, NvrtcSourceProvenance.CallerProvidedRaw)
    assertEquals(diagnostic.generatedLocation.file, "raw_kernel.cu")
    assertEquals(diagnostic.generatedLocation.line, 1)
    assertEquals(diagnostic.generatedLocation.column, Some(36))
    assertEquals(diagnostic.sourceSpan, None)
