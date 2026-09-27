package flight4s.runtime.cuda

import flight4s.core.codegen.GeneratedCudaModule
import flight4s.core.compiler.*
import flight4s.core.unsafe.raw.RawCudaKernel
import flight4s.runtime.cuda.internal.NativeNvrtcCompiler

private[cuda] trait NvrtcCompilerBackend:
  def version(): Either[NvrtcVersionQueryFailure, NvrtcVersion]

  def compile(
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompileFailure, NvrtcArtifact]

private[cuda] object NativeNvrtcCompilerBackend extends NvrtcCompilerBackend:
  override def version(): Either[NvrtcVersionQueryFailure, NvrtcVersion] =
    val nativeResult = NativeNvrtcCompiler.version()
    if nativeResult.resultCode == 0 then
      Right(
        NvrtcVersion(
          nativeResult.versionMajor,
          nativeResult.versionMinor
        )
      )
    else
      Left(
        NvrtcVersionQueryFailure(
          nativeResult.resultCode,
          nativeResult.resultName
        )
      )

  override def compile(
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompileFailure, NvrtcArtifact] =
    NvrtcCompiler.validateRequest(input, programName)
    val compilerOptions = NvrtcCompileOptions.resolve(
      input.compilerOptions,
      target
    )
    val nativeResult = NativeNvrtcCompiler.compile(
      input.source,
      programName,
      compilerOptions.values
    )
    val version = NvrtcVersion(
      nativeResult.versionMajor,
      nativeResult.versionMinor
    )

    if nativeResult.resultCode == 0 then
      Right(
        NvrtcArtifact(
          input = input,
          ptx = IArray.unsafeFromArray(nativeResult.ptx.clone()),
          compileLog = nativeResult.compileLog,
          nvrtcVersion = version,
          target = target,
          compilerOptions = compilerOptions,
          programName = programName
        )
      )
    else
      Left(
        NvrtcCompileFailure(
          input = input,
          resultCode = nativeResult.resultCode,
          resultName = nativeResult.resultName,
          compileLog = nativeResult.compileLog,
          nvrtcVersion = version,
          target = target,
          compilerOptions = compilerOptions,
          programName = programName
        )
      )

object NvrtcCompiler:
  val DefaultProgramName: String = "flight4s_generated.cu"

  def version(): Either[NvrtcVersionQueryFailure, NvrtcVersion] =
    NativeNvrtcCompilerBackend.version()

  def compile(
      input: NvrtcCompilationInput,
      target: ComputeCapability
  ): Either[NvrtcCompileFailure, NvrtcArtifact] =
    compile(input, target, DefaultProgramName)

  def compile(
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompileFailure, NvrtcArtifact] =
    NativeNvrtcCompilerBackend.compile(input, target, programName)

  def compile(
      generated: GeneratedCudaModule,
      target: ComputeCapability
  ): Either[NvrtcCompileFailure, NvrtcArtifact] =
    compile(generated, target, DefaultProgramName)

  def compile(
      generated: GeneratedCudaModule,
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompileFailure, NvrtcArtifact] =
    compile(
      NvrtcCompilationInput.generated(generated),
      target,
      programName
    )

  def compile[Args <: Tuple](
      raw: RawCudaKernel[Args],
      target: ComputeCapability
  ): Either[NvrtcCompileFailure, NvrtcArtifact] =
    compile(raw, target, DefaultProgramName)

  def compile[Args <: Tuple](
      raw: RawCudaKernel[Args],
      target: ComputeCapability,
      programName: String
  ): Either[NvrtcCompileFailure, NvrtcArtifact] =
    compile(NvrtcCompilationInput.raw(raw), target, programName)

  private[cuda] def validateRequest(
      input: NvrtcCompilationInput,
      programName: String
  ): Unit =
    require(
      input != null,
      "NVRTC compilation input must not be null"
    )
    require(
      input.source != null && input.source.nonEmpty,
      "CUDA source must not be empty"
    )
    require(
      programName.nonEmpty,
      "NVRTC program name must not be empty"
    )
    require(
      !programName.contains('\u0000'),
      "NVRTC program name must not contain a null character"
    )
