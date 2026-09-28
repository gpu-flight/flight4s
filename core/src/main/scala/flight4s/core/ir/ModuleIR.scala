package flight4s.core.ir

final case class CudaModuleIR(
    constants: Vector[ConstantArray[?]],
    kernels: Vector[KernelIR[?]]
)

object ModuleValidator:

  def validate(module: CudaModuleIR): ValidationResult =
    val constantErrors = module.constants.zipWithIndex.flatMap {
      case (constant, index) =>
        val location = s"constants[$index]"
        val nameErrors =
          if CudaIdentifier.isValid(constant.name) then Vector.empty
          else
            Vector(
              ValidationError(
                ValidationCode.InvalidConstantName,
                s"'${constant.name}' is not a valid CUDA constant identifier",
                location,
                constant.span
              )
            )
        val sizeErrors =
          if constant.elementCount > 0 then Vector.empty
          else
            Vector(
              ValidationError(
                ValidationCode.InvalidConstantElementCount,
                s"constant array '${constant.name}' must have a positive element count",
                location,
                constant.span
              )
            )

        nameErrors ++ sizeErrors
    }

    val duplicateConstantErrors = module.constants
      .groupBy(_.name)
      .collect { case (name, declarations) if declarations.size > 1 =>
        ValidationError(
          ValidationCode.DuplicateConstantName,
          s"constant array '$name' is declared ${declarations.size} times",
          "constants"
        )
      }
      .toVector
      .sortBy(_.message)

    val duplicateKernelErrors = module.kernels
      .groupBy(_.name)
      .collect { case (name, declarations) if declarations.size > 1 =>
        ValidationError(
          ValidationCode.DuplicateKernelName,
          s"kernel '$name' is declared ${declarations.size} times",
          "kernels"
        )
      }
      .toVector
      .sortBy(_.message)

    val constantNames = module.constants.map(_.name).toSet
    val kernelNames = module.kernels.map(_.name).toSet
    val symbolConflictErrors = constantNames
      .intersect(kernelNames)
      .toVector
      .sorted
      .map { name =>
        ValidationError(
          ValidationCode.ModuleSymbolConflict,
          s"module symbol '$name' is used by both a constant and a kernel",
          "module"
        )
      }

    val kernelValidationResults = module.kernels.zipWithIndex.map {
      case (kernel, index) =>
        KernelValidator
          .validateInModule(kernel, module.constants)
    }

    val kernelErrors = kernelValidationResults.zipWithIndex.flatMap {
      case (result, index) =>
        result.errors.map(error => error.copy(location = s"kernels[$index].${error.location}"))
    }

    val kernelWarnings = kernelValidationResults.zipWithIndex.flatMap {
      case (result, index) =>
        result.warnings.map(warning =>
          warning.copy(location = s"kernels[$index].${warning.location}")
        )
    }

    ValidationResult(
      constantErrors ++
        duplicateConstantErrors ++
        duplicateKernelErrors ++
        symbolConflictErrors ++
        kernelErrors,
      kernelWarnings
    )
