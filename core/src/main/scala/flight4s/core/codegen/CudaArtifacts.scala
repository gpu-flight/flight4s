package flight4s.core.codegen

import flight4s.core.ir.{KernelSignature, SourceSpan, ValidationError}
import flight4s.core.launch.{Block as LaunchBlock}

enum CudaCppStandard(val nvrtcOption: String):
  case Cpp20 extends CudaCppStandard("--std=c++20")

final case class CompilerOptions(
    languageStandard: CudaCppStandard = CudaCppStandard.Cpp20,
    additionalNvrtcOptions: Vector[String] = Vector.empty
):
  require(
    additionalNvrtcOptions.forall { option =>
      option.nonEmpty &&
      !option.contains('\u0000') &&
      !option.startsWith("--std") &&
      !option.startsWith("-std") &&
      !option.startsWith("--gpu-architecture") &&
      !option.startsWith("-arch")
    },
    "additional NVRTC options must be non-empty, null-free, and cannot " +
      "override the CUDA C++ standard or target"
  )

  def nvrtcOptions: Vector[String] =
    languageStandard.nvrtcOption +: additionalNvrtcOptions

final case class SourceMapEntry(
    generatedLine: Int,
    sourceSpan: SourceSpan
)

final case class SourceMap(entries: Vector[SourceMapEntry]):
  def closestAtOrBefore(generatedLine: Int): Option[SourceMapEntry] =
    entries
      .filter(_.generatedLine <= generatedLine)
      .maxByOption(_.generatedLine)

final case class DynamicSharedMemoryRequirement(
    elementSizeBytes: Int,
    elementAlignmentBytes: Int
):
  require(elementSizeBytes > 0, "element size must be positive")
  require(elementAlignmentBytes > 0, "element alignment must be positive")

final case class KernelLaunchRequirements(
    dynamicSharedMemory: Option[DynamicSharedMemoryRequirement] = None,
    requiredBlock: Option[LaunchBlock] = None
)

final case class GeneratedKernel[Args <: Tuple](
    name: String,
    signature: KernelSignature[Args],
    cudaSource: String,
    sourceMap: SourceMap,
    compilerOptions: CompilerOptions,
    declarationLine: Int,
    launchRequirements: KernelLaunchRequirements
)

final case class GeneratedCudaModule(
    cudaSource: String,
    sourceMap: SourceMap,
    compilerOptions: CompilerOptions,
    kernels: Vector[GeneratedKernel[?]]
)

sealed trait CodegenError:
  def message: String

object CodegenError:
  final case class ValidationFailed(errors: Vector[ValidationError])
      extends CodegenError:
    override val message: String =
      s"CUDA source generation rejected ${errors.size} validation error(s)"

  final case class UnsupportedLiteral(
      cudaType: String,
      valueClass: String,
      span: SourceSpan
  ) extends CodegenError:
    override val message: String =
      s"cannot emit a $cudaType literal from $valueClass"

  final case class UnsupportedConversion(
      fromType: String,
      toType: String,
      span: SourceSpan
  ) extends CodegenError:
    override val message: String =
      s"cannot emit a conversion from $fromType to $toType"
