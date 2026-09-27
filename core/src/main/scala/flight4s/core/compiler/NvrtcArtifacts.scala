package flight4s.core.compiler

import flight4s.core.codegen.CompilerOptions

final case class ComputeCapability private (
    major: Int,
    minor: Int
):
  require(major > 0, s"compute capability major must be positive: $major")
  require(
    minor >= 0 && minor <= 9,
    s"compute capability minor must be between 0 and 9: $minor"
  )

  def virtualArchitecture: String =
    s"compute_$major$minor"

  def nvrtcOption: String =
    s"--gpu-architecture=$virtualArchitecture"

  override def toString: String =
    s"$major.$minor"

object ComputeCapability:
  def apply(major: Int, minor: Int): ComputeCapability =
    new ComputeCapability(major, minor)

final case class NvrtcVersion(
    major: Int,
    minor: Int
):
  require(major >= 0, s"NVRTC version major must not be negative: $major")
  require(minor >= 0, s"NVRTC version minor must not be negative: $minor")

  override def toString: String =
    s"$major.$minor"

sealed trait NvrtcCompilationError:
  def message: String

final case class NvrtcVersionQueryFailure(
    resultCode: Int,
    resultName: String
) extends NvrtcCompilationError:
  require(resultCode != 0, "an NVRTC version failure must have a nonzero code")
  require(resultName.nonEmpty, "an NVRTC version failure must have a name")

  def message: String =
    s"NVRTC version query failed with $resultName ($resultCode)"

final case class NvrtcCompileOptions private (
    values: Vector[String]
)

object NvrtcCompileOptions:
  def resolve(
      generatedOptions: CompilerOptions,
      target: ComputeCapability
  ): NvrtcCompileOptions =
    NvrtcCompileOptions(
      generatedOptions.nvrtcOptions :+ target.nvrtcOption
    )

  /** Rebuilds a previously resolved option sequence from a trusted artifact. */
  private[flight4s] def fromResolvedValues(
      values: Vector[String]
  ): NvrtcCompileOptions =
    require(values.nonEmpty, "resolved NVRTC options must not be empty")
    require(
      values.forall(option => option.nonEmpty && !option.contains('\u0000')),
      "resolved NVRTC options must be non-empty and null-free"
    )
    new NvrtcCompileOptions(values)

sealed trait NvrtcCompilation:
  def input: NvrtcCompilationInput
  def compileLog: String
  def nvrtcVersion: NvrtcVersion
  def target: ComputeCapability
  def compilerOptions: NvrtcCompileOptions
  def programName: String

  final def provenance: NvrtcSourceProvenance = input.provenance

  final def diagnostics: Vector[NvrtcDiagnostic] =
    NvrtcDiagnostics.parse(
      compileLog,
      input.sourceMap,
      programName
    )

final case class NvrtcArtifact(
    input: NvrtcCompilationInput,
    ptx: IArray[Byte],
    compileLog: String,
    nvrtcVersion: NvrtcVersion,
    target: ComputeCapability,
    compilerOptions: NvrtcCompileOptions,
    programName: String
) extends NvrtcCompilation:
  require(ptx.nonEmpty, "successful NVRTC compilation must contain PTX")

final case class NvrtcCompileFailure(
    input: NvrtcCompilationInput,
    resultCode: Int,
    resultName: String,
    compileLog: String,
    nvrtcVersion: NvrtcVersion,
    target: ComputeCapability,
    compilerOptions: NvrtcCompileOptions,
    programName: String
) extends NvrtcCompilation,
      NvrtcCompilationError:
  def message: String =
    s"NVRTC compilation failed with $resultName ($resultCode)"
