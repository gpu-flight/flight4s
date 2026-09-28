package flight4s.core.ir

import flight4s.core.types.{BF16, Bool, CudaType, F16, F32, F64, FP8E4M3, FP8E5M2, I32, U32, UInt}

enum ValidationCode:
  case InvalidConstantName
  case DuplicateConstantName
  case InvalidConstantElementCount
  case ConstantNameShadowed
  case ModuleContextRequired
  case DuplicateKernelName
  case ModuleSymbolConflict
  case InvalidKernelName
  case InvalidParameterName
  case DuplicateParameterName
  case InvalidSharedMemoryName
  case DuplicateSharedMemoryName
  case SharedMemoryNameConflictsWithParameter
  case InvalidSharedMemoryElementCount
  case InvalidSharedMemoryLayout
  case MultipleDynamicSharedDeclarations
  case InvalidLocalName
  case DuplicateLocalName
  case LocalNameConflictsWithBinding
  case InvalidLocalArrayElementCount
  case UnboundLocal
  case UnknownLocalArray
  case LocalArrayTypeMismatch
  case LocalArrayIndexOutOfBounds
  case LocalTypeMismatch
  case InvalidLoopIndexName
  case InvalidLoopStep
  case LoopIndexConflictsWithBinding
  case UnboundLoopIndex
  case UnknownBuffer
  case UnknownScalarParameter
  case ExpectedBuffer
  case ExpectedScalarParameter
  case BufferTypeMismatch
  case WriteToReadOnlyBuffer
  case InvalidAtomicAddressSpace
  case EmptyWarpMask
  case InvalidWarpWidth
  case NegativeWarpSourceLane
  case InvalidWarpSelector
  case UnknownConstant
  case ConstantTypeMismatch
  case ConstantIndexOutOfBounds
  case UnknownSharedMemory
  case SharedMemoryTypeMismatch
  case SharedMemoryIndexRankMismatch
  case SharedMemoryIndexOutOfBounds
  case ExpressionTypeMismatch
  case UnsupportedBitwiseType
  case UnsupportedConversion
  case UnsupportedConversionRounding
  case UnsupportedConversionSaturation
  case UnknownIntrinsic
  case InvalidReductionIndexName
  case DuplicateReductionIndex
  case ReductionIndexConflictsWithParameter
  case UnboundReductionIndex

enum ValidationWarningCode:
  case BarrierMayDiverge

final case class ValidationError(
    code: ValidationCode,
    message: String,
    location: String,
    span: SourceSpan = SourceSpan.Unknown
)

final case class ValidationWarning(
    code: ValidationWarningCode,
    message: String,
    location: String,
    span: SourceSpan = SourceSpan.Unknown
)

final case class ValidationResult(
    errors: Vector[ValidationError],
    warnings: Vector[ValidationWarning] = Vector.empty
):
  def isValid: Boolean = errors.isEmpty

  def toEither: Either[Vector[ValidationError], Unit] =
    if isValid then Right(()) else Left(errors)

object KernelValidator:
  private final case class LocalArrayDeclarationValidation(
      errors: Vector[ValidationError],
      bindingErrors: Vector[ValidationError]
  )

  private final case class ValidationScope(
      locals: Map[String, CudaType[?]] = Map.empty,
      localArrays: Map[String, LocalArray[?]] = Map.empty,
      loopIndexes: Set[String] = Set.empty,
      reductionIndexes: Set[String] = Set.empty,
      constants: Option[Map[String, ConstantArray[?]]] = None,
      sharedMemory: Map[String, SharedArray[?, ?]] = Map.empty
  )

  private val cudaIdentifier = raw"[A-Za-z_][A-Za-z0-9_]*".r
  private val intrinsicTypes: Map[String, CudaType[?]] = Map(
    "threadIdx.x" -> I32,
    "threadIdx.y" -> I32,
    "threadIdx.z" -> I32,
    "blockIdx.x" -> I32,
    "blockIdx.y" -> I32,
    "blockIdx.z" -> I32,
    "blockDim.x" -> I32,
    "blockDim.y" -> I32,
    "blockDim.z" -> I32
  )

  def validate(kernel: Kernel[?]): ValidationResult =
    validate(kernel.ir)

  def validate(kernel: KernelIR[?]): ValidationResult =
    validateKernel(kernel, None)

  private[ir] def validateInModule(
      kernel: KernelIR[?],
      constants: Vector[ConstantArray[?]]
  ): ValidationResult =
    validateKernel(kernel, Some(constants))

  private def validateKernel(
      kernel: KernelIR[?],
      constants: Option[Vector[ConstantArray[?]]]
  ): ValidationResult =
    val constantNames = constants.fold(Set.empty[String])(_.map(_.name).toSet)
    val parameterErrors = validateParameters(kernel, constantNames)
    val sharedMemoryErrors = validateSharedMemory(kernel, constantNames)
    val parametersByName = kernel.params.groupBy(_.name).view.mapValues(_.head).toMap
    val bodyErrors = validateBlock(
      kernel.body,
      parametersByName,
      "body",
      ValidationScope(
        constants = constants.map(
          _.groupBy(_.name).view.mapValues(_.head).toMap
        ),
        sharedMemory =
          kernel.sharedMemory.groupBy(_.name).view.mapValues(_.head).toMap
      )
    )

    ValidationResult(
      parameterErrors ++ sharedMemoryErrors ++ bodyErrors,
      BarrierDivergenceAnalysis.warnings(kernel.body)
    )

  private def validateParameters(
      kernel: KernelIR[?],
      constantNames: Set[String]
  ): Vector[ValidationError] =
    val kernelNameErrors =
      if isIdentifier(kernel.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidKernelName,
            s"'${kernel.name}' is not a valid CUDA kernel identifier",
            "kernel"
          )
        )

    val nameErrors = kernel.params.zipWithIndex.flatMap { case (parameter, index) =>
      if isIdentifier(parameter.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidParameterName,
            s"'${parameter.name}' is not a valid CUDA parameter identifier",
            s"params[$index]"
          )
        )
    }

    val duplicateErrors = kernel.params
      .groupBy(_.name)
      .collect { case (name, parameters) if parameters.size > 1 =>
        ValidationError(
          ValidationCode.DuplicateParameterName,
          s"CUDA parameter '$name' is declared ${parameters.size} times",
          "params"
        )
      }
      .toVector
      .sortBy(_.message)

    val constantShadowErrors = kernel.params.zipWithIndex.flatMap {
      case (parameter, index) =>
        if constantNames.contains(parameter.name) then
          Vector(
            ValidationError(
              ValidationCode.ConstantNameShadowed,
              s"parameter '${parameter.name}' shadows a module constant",
              s"params[$index]"
            )
          )
        else Vector.empty
    }

    kernelNameErrors ++ nameErrors ++ duplicateErrors ++ constantShadowErrors

  private def validateSharedMemory(
      kernel: KernelIR[?],
      constantNames: Set[String]
  ): Vector[ValidationError] =
    val parameterNames = kernel.params.map(_.name).toSet
    val nameErrors = kernel.sharedMemory.zipWithIndex.flatMap {
      case (memory, index) =>
        val location = s"sharedMemory[$index]"
        val identifierErrors =
          if isIdentifier(memory.name) then Vector.empty
          else
            Vector(
              ValidationError(
                ValidationCode.InvalidSharedMemoryName,
                s"'${memory.name}' is not a valid CUDA shared-memory identifier",
                location,
                memory.span
              )
            )
        val parameterConflictErrors =
          if parameterNames.contains(memory.name) then
            Vector(
              ValidationError(
                ValidationCode.SharedMemoryNameConflictsWithParameter,
                s"shared memory '${memory.name}' conflicts with a parameter",
                location,
                memory.span
              )
            )
          else Vector.empty
        val constantShadowErrors =
          if constantNames.contains(memory.name) then
            Vector(
              ValidationError(
                ValidationCode.ConstantNameShadowed,
                s"shared memory '${memory.name}' shadows a module constant",
                location,
                memory.span
              )
            )
          else Vector.empty
        val sizeErrors = memory.size match
          case static: StaticSharedMemory[?] =>
            validateSharedMemoryLayout(
              memory,
              static.layout,
              static.rank,
              location
            )
          case DynamicSharedMemory =>
            Vector.empty

        identifierErrors ++
          parameterConflictErrors ++
          constantShadowErrors ++
          sizeErrors
    }

    val duplicateErrors = kernel.sharedMemory
      .groupBy(_.name)
      .collect { case (name, declarations) if declarations.size > 1 =>
        ValidationError(
          ValidationCode.DuplicateSharedMemoryName,
          s"shared memory '$name' is declared ${declarations.size} times",
          "sharedMemory"
        )
      }
      .toVector
      .sortBy(_.message)

    val dynamicDeclarations =
      kernel.sharedMemory.count(_.size == DynamicSharedMemory)
    val dynamicErrors =
      if dynamicDeclarations <= 1 then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.MultipleDynamicSharedDeclarations,
            s"a kernel may declare at most one dynamic shared array, found $dynamicDeclarations",
            "sharedMemory"
          )
        )

    nameErrors ++ duplicateErrors ++ dynamicErrors

  private def validateSharedMemoryLayout(
      memory: SharedArray[?, ?],
      layout: MemoryLayout[?],
      declaredRank: Int,
      location: String
  ): Vector[ValidationError] =
    val positiveDimensionErrors =
      if layout.logicalDimensions.forall(_ > 0) &&
          layout.physicalDimensions.forall(_ > 0)
      then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidSharedMemoryElementCount,
            s"shared memory '${memory.name}' dimensions must be positive",
            location,
            memory.span
          )
        )

    val supportedRank =
      (layout.rank >= 1 && layout.rank <= 3) &&
        layout.rank == declaredRank &&
        declaredRank == memory.rankWitness.rank
    val matchingRanks =
      layout.logicalDimensions.size == layout.physicalDimensions.size
    val physicalContainsLogical =
      matchingRanks &&
        layout.logicalDimensions
          .zip(layout.physicalDimensions)
          .forall { case (logical, physical) => physical >= logical }
    val onlyInnermostDimensionIsPadded =
      matchingRanks &&
        layout.logicalDimensions.dropRight(1) ==
          layout.physicalDimensions.dropRight(1)

    val layoutErrors =
      if supportedRank &&
          matchingRanks &&
          physicalContainsLogical &&
          onlyInnermostDimensionIsPadded
      then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidSharedMemoryLayout,
            s"shared memory '${memory.name}' must have a supported row-major layout " +
              "whose physical innermost dimension contains its logical dimension",
            location,
            memory.span
          )
        )

    positiveDimensionErrors ++ layoutErrors

  private def validateBlock(
      block: Block,
      parameters: Map[String, KernelParam],
      location: String,
      initialScope: ValidationScope = ValidationScope()
  ): Vector[ValidationError] =
    block.statements.zipWithIndex
      .foldLeft((Vector.empty[ValidationError], initialScope)) {
        case ((errors, scope), (declaration: LocalDeclaration[?], index)) =>
          val statementLocation = s"$location.statements[$index]"
          val nameErrors =
            validateLocalName(declaration.local, parameters, scope, statementLocation)
          val declarationErrors =
            nameErrors ++
              validateExpression(
                declaration.initial,
                parameters,
                s"$statementLocation.initial",
                scope
              ) ++
              requireSameType(
                declaration.initial.valueType,
                declaration.local.valueType,
                "local initializer type does not match the declared local type",
                s"$statementLocation.initial",
                declaration.initial.span,
                ValidationCode.LocalTypeMismatch
              )
          val nextScope =
            if nameErrors.isEmpty then
              scope.copy(
                locals = scope.locals.updated(
                  declaration.local.name,
                  declaration.local.valueType
                )
              )
            else scope

          (errors ++ declarationErrors, nextScope)

        case ((errors, scope), (shuffle: WarpShuffle[?, ?], index)) =>
          val statementLocation = s"$location.statements[$index]"
          val nameErrors = validateLocalName(shuffle.local, parameters, scope, statementLocation)
          val declarationErrors = nameErrors ++ validateWarpShuffle(shuffle, parameters, statementLocation, scope)
          val nextScope =
            if nameErrors.isEmpty then scope.copy(
              locals = scope.locals.updated(shuffle.local.name, shuffle.local.valueType)
            )
            else scope
          (errors ++ declarationErrors, nextScope)

        case ((errors, scope), (vote: WarpVote[?], index)) =>
          val statementLocation = s"$location.statements[$index]"
          val nameErrors = validateLocalName(vote.local, parameters, scope, statementLocation)
          val declarationErrors = nameErrors ++ validateWarpVote(vote, parameters, statementLocation, scope)
          val nextScope =
            if nameErrors.isEmpty then scope.copy(
              locals = scope.locals.updated(vote.local.name, vote.local.valueType)
            )
            else scope
          (errors ++ declarationErrors, nextScope)

        case ((errors, scope), (atomic: AtomicFetchAdd[?, ?], index)) =>
          val statementLocation = s"$location.statements[$index]"
          val nameErrors = validateLocalName(atomic.local, parameters, scope, statementLocation)
          val declarationErrors = nameErrors ++
            validateAtomicAdd(atomic.target, atomic.value, atomic.addition, atomic.span,
              parameters, statementLocation, scope) ++
            requireSameType(atomic.local.valueType, atomic.target.valueType,
              "atomic result type does not match the target type", statementLocation,
              atomic.span, ValidationCode.LocalTypeMismatch)
          val nextScope =
            if nameErrors.isEmpty then scope.copy(
              locals = scope.locals.updated(atomic.local.name, atomic.local.valueType)
            )
            else scope
          (errors ++ declarationErrors, nextScope)

        case ((errors, scope), (declaration: LocalArrayDeclaration[?], index)) =>
          val statementLocation = s"$location.statements[$index]"
          val declarationValidation =
            validateLocalArrayDeclaration(
              declaration.array,
              parameters,
              scope,
              statementLocation
            )
          val nextScope =
            if declarationValidation.bindingErrors.isEmpty then
              scope.copy(
                localArrays =
                  scope.localArrays.updated(
                    declaration.array.name,
                    declaration.array
                  )
              )
            else scope

          (errors ++ declarationValidation.errors, nextScope)

        case ((errors, scope), (statement: ExecutableStmt, index)) =>
          (
            errors ++ validateStatement(
              statement,
              parameters,
              s"$location.statements[$index]",
              scope
            ),
            scope
          )
      }
      ._1

  private def validateStatement(
      statement: ExecutableStmt,
      parameters: Map[String, KernelParam],
      location: String,
      scope: ValidationScope
  ): Vector[ValidationError] =
    statement match
      case store: Store[?, ?] =>
        validatePlace(
          store.to,
          parameters,
          s"$location.to",
          isWrite = true,
          scope = scope
        ) ++
          validateExpression(store.value, parameters, s"$location.value", scope) ++
          requireSameType(
            store.to.valueType,
            store.value.valueType,
            s"store target type ${store.to.valueType.cudaName} does not match " +
              s"value type ${store.value.valueType.cudaName}",
            location,
            store.span
          )

      case atomic: AtomicAdd[?, ?] =>
        validateAtomicAdd(atomic.target, atomic.value, atomic.addition, atomic.span,
          parameters, location, scope)

      case accumulation: Accumulate[?] =>
        validatePlace(
          accumulation.target,
          parameters,
          s"$location.target",
          isWrite = true,
          scope = scope
        ) ++
          validateExpression(
            accumulation.value,
            parameters,
            s"$location.value",
            scope
          ) ++
          requireSameType(
            accumulation.target.valueType,
            accumulation.value.valueType,
            "accumulated value type does not match the local accumulator type",
            s"$location.value",
            accumulation.value.span
          ) ++
          requireSameType(
            accumulation.addition,
            accumulation.target.valueType,
            "accumulation capability does not match the local accumulator type",
            location,
            accumulation.span
          )

      case branch: IfThen =>
        validateExpression(
          branch.condition,
          parameters,
          s"$location.condition",
          scope
        ) ++
          requireSameType(
            branch.condition.valueType,
            Bool,
            "GPU branch condition must have CUDA bool type",
            s"$location.condition",
            branch.condition.span
          ) ++
          validateBlock(branch.thenBlock, parameters, s"$location.then", scope) ++
          branch.elseBlock.toVector.flatMap(
            validateBlock(_, parameters, s"$location.else", scope)
          )

      case scoped: ScopedBlock =>
        validateBlock(scoped.body, parameters, s"$location.body", scope)

      case loop: ForLoop =>
        val nameErrors =
          validateLoopIndexName(loop.index, parameters, scope, s"$location.index")
        val stepErrors =
          validateLoopStep(loop.step, location, loop.span)
        val rangeErrors =
          validateExpression(loop.from, parameters, s"$location.from", scope) ++
            validateExpression(loop.until, parameters, s"$location.until", scope) ++
            requireSameType(
              loop.from.valueType,
              I32,
              "loop lower bound must have CUDA int type",
              s"$location.from",
              loop.from.span
            ) ++
            requireSameType(
              loop.until.valueType,
              I32,
              "loop upper bound must have CUDA int type",
              s"$location.until",
              loop.until.span
            )
        val bodyScope =
          scope.copy(loopIndexes = scope.loopIndexes + loop.index.name)

        nameErrors ++ rangeErrors ++ stepErrors ++
          validateBlock(loop.body, parameters, s"$location.body", bodyScope)

      case _: Barrier =>
        Vector.empty

      case barrier: WarpBarrier =>
        validateNonemptyWarpMask(barrier.mask, location, barrier.span) ++
          validateExpression(barrier.mask, parameters, s"$location.mask", scope) ++
          requireSameType(barrier.mask.valueType, U32, "warp synchronization mask must have CUDA unsigned int type",
            s"$location.mask", barrier.span)

  private def validateLoopStep(step: Int, location: String, span: SourceSpan): Vector[ValidationError] =
    if step > 0 then Vector.empty
    else Vector(ValidationError(ValidationCode.InvalidLoopStep,
      "loop step must be a positive static integer", s"$location.step", span))

  private def validateWarpShuffle(
      shuffle: WarpShuffle[?, ?],
      parameters: Map[String, KernelParam],
      location: String,
      scope: ValidationScope
  ): Vector[ValidationError] =
    val widthErrors =
      if Set(1, 2, 4, 8, 16, 32).contains(shuffle.width) then Vector.empty
      else Vector(ValidationError(ValidationCode.InvalidWarpWidth,
        "warp shuffle width must be one of 1, 2, 4, 8, 16, 32", location, shuffle.span))
    val selectorLocation = s"$location.${shuffle.operator.selectorName}"
    val selector: Expr[?] = shuffle.selector
    val laneErrors = (shuffle.operator, selector) match
      case (WarpShuffleOperator.Direct, Literal(value: Int, _, _)) if value < 0 =>
        Vector(ValidationError(ValidationCode.NegativeWarpSourceLane,
          "warp shuffle source lane must be nonnegative", selectorLocation, shuffle.span))
      case (WarpShuffleOperator.Xor, Literal(value: Int, _, _)) if value < 0 || value > 31 =>
        Vector(ValidationError(ValidationCode.InvalidWarpSelector,
          "warp shuffle lane mask must be in 0..31", selectorLocation, shuffle.span))
      case (WarpShuffleOperator.Up | WarpShuffleOperator.Down, Literal(value: UInt, _, _))
          if value.toIntBits < 0 || value.toIntBits > 31 =>
        Vector(ValidationError(ValidationCode.InvalidWarpSelector,
          "warp shuffle delta must be in 0..31", selectorLocation, shuffle.span))
      case _ => Vector.empty
    widthErrors ++ laneErrors ++
      validateNonemptyWarpMask(shuffle.mask, location, shuffle.span) ++
      validateExpression(shuffle.mask, parameters, s"$location.mask", scope) ++
      validateExpression(shuffle.value, parameters, s"$location.value", scope) ++
      validateExpression(shuffle.selector, parameters, selectorLocation, scope) ++
      requireSameType(shuffle.mask.valueType, U32, "warp shuffle mask must have CUDA unsigned int type",
        s"$location.mask", shuffle.span) ++
      requireSameType(shuffle.selector.valueType, shuffle.operator.selectorType,
        s"warp shuffle ${shuffle.operator.selectorName} must have CUDA ${shuffle.operator.selectorType.cudaName} type",
        selectorLocation, shuffle.span) ++
      requireSameType(shuffle.local.valueType, shuffle.value.valueType,
        "warp shuffle result type does not match the value", location, shuffle.span, ValidationCode.LocalTypeMismatch) ++
      requireSameType(shuffle.shuffleType, shuffle.value.valueType,
        "warp shuffle capability does not match the value type", location, shuffle.span)

  private def validateWarpVote(
      vote: WarpVote[?],
      parameters: Map[String, KernelParam],
      location: String,
      scope: ValidationScope
  ): Vector[ValidationError] =
    validateNonemptyWarpMask(vote.mask, location, vote.span) ++
      validateExpression(vote.mask, parameters, s"$location.mask", scope) ++
      validateExpression(vote.predicate, parameters, s"$location.predicate", scope) ++
      requireSameType(vote.mask.valueType, U32, "warp vote mask must have CUDA unsigned int type",
        s"$location.mask", vote.span) ++
      requireSameType(vote.predicate.valueType, Bool, "warp vote predicate must have CUDA bool type",
        s"$location.predicate", vote.span) ++
      requireSameType(vote.local.valueType, vote.operator.resultType,
        "warp vote result type does not match the operator", location, vote.span, ValidationCode.LocalTypeMismatch)

  private def validateNonemptyWarpMask(
      mask: Expr[?],
      location: String,
      span: SourceSpan
  ): Vector[ValidationError] = mask match
      case Literal(value: UInt, _, _) if value.toIntBits == 0 =>
        Vector(ValidationError(ValidationCode.EmptyWarpMask,
          "a calling lane must belong to a nonempty warp participation mask", s"$location.mask", span))
      case _ => Vector.empty

  private def validateAtomicAdd(
      target: Place[?, ?, ?],
      value: Expr[?],
      addition: CudaType[?],
      span: SourceSpan,
      parameters: Map[String, KernelParam],
      location: String,
      scope: ValidationScope
  ): Vector[ValidationError] =
    val spaceErrors = target match
      case _: BufferElement[?, ?] | _: SharedElement[?] => Vector.empty
      case _ => Vector(ValidationError(
        ValidationCode.InvalidAtomicAddressSpace,
        "atomicAdd requires a global or shared memory target",
        s"$location.target",
        span
      ))
    spaceErrors ++
      validatePlace(target, parameters, s"$location.target", isWrite = true, scope = scope) ++
      validateExpression(value, parameters, s"$location.value", scope) ++
      requireSameType(target.valueType, value.valueType,
        "atomicAdd value type does not match the target type", s"$location.value", span) ++
      requireSameType(addition, target.valueType,
        "atomicAdd capability does not match the target type", location, span)

  private def validateExpression(
      expression: Expr[?],
      parameters: Map[String, KernelParam],
      location: String,
      scope: ValidationScope = ValidationScope()
  ): Vector[ValidationError] =
    expression match
      case _: Literal[?] =>
        Vector.empty

      case scalar: ScalarParam[?] =>
        parameters.get(scalar.name) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnknownScalarParameter,
                s"scalar parameter '${scalar.name}' is not declared",
                location,
                scalar.span
              )
            )

          case Some(_: BufferParam[?, ?]) =>
            Vector(
              ValidationError(
                ValidationCode.ExpectedScalarParameter,
                s"parameter '${scalar.name}' is a buffer, not a scalar",
                location,
                scalar.span
              )
            )

          case Some(declared: ScalarParam[?]) =>
            requireSameType(
              scalar.valueType,
              declared.valueType,
              s"scalar parameter '${scalar.name}' has type ${scalar.valueType.cudaName}, " +
                s"but was declared as ${declared.valueType.cudaName}",
              location,
              scalar.span
            )

      case binary: Binary[?] =>
        val operatorErrors = binary.operator match
          case BinaryOperator.BitAnd | BinaryOperator.BitOr | BinaryOperator.BitXor
              if binary.valueType != I32 && binary.valueType != U32 =>
            Vector(ValidationError(ValidationCode.UnsupportedBitwiseType,
              "bitwise operands must have CUDA int or unsigned int type", location, binary.span))
          case _ => Vector.empty
        validateExpression(binary.left, parameters, s"$location.left", scope) ++
          validateExpression(binary.right, parameters, s"$location.right", scope) ++
          requireSameType(
            binary.left.valueType,
            binary.valueType,
            "left operand type does not match binary result type",
            s"$location.left",
            binary.left.span
          ) ++
          requireSameType(
            binary.right.valueType,
            binary.valueType,
            "right operand type does not match binary result type",
            s"$location.right",
            binary.right.span
          ) ++ operatorErrors

      case shift: UnsignedShift =>
        validateExpression(shift.value, parameters, s"$location.value", scope) ++
          validateExpression(shift.distance, parameters, s"$location.distance", scope) ++
          requireSameType(shift.value.valueType, U32,
            "unsigned shift value must have CUDA unsigned int type", s"$location.value", shift.value.span) ++
          requireSameType(shift.distance.valueType, I32,
            "unsigned shift distance must have CUDA int type", s"$location.distance", shift.distance.span)

      case shift: SignedShift =>
        validateExpression(shift.value, parameters, s"$location.value", scope) ++
          validateExpression(shift.distance, parameters, s"$location.distance", scope) ++
          requireSameType(shift.value.valueType, I32,
            "signed shift value must have CUDA int type", s"$location.value", shift.value.span) ++
          requireSameType(shift.distance.valueType, I32,
            "signed shift distance must have CUDA int type", s"$location.distance", shift.distance.span)

      case count: PopulationCount[?] =>
        validateExpression(count.value, parameters, s"$location.value", scope) ++
          requireSameType(count.value.valueType, count.wordType,
            "population-count operand type does not match its word type", s"$location.value", count.value.span)

      case comparison: Compare[?] =>
        validateExpression(
          comparison.left,
          parameters,
          s"$location.left",
          scope
        ) ++
          validateExpression(
            comparison.right,
            parameters,
            s"$location.right",
            scope
          ) ++
          requireSameType(
            comparison.left.valueType,
            comparison.operandType,
            "left operand type does not match comparison operand type",
            s"$location.left",
            comparison.left.span
          ) ++
          requireSameType(
            comparison.right.valueType,
            comparison.operandType,
            "right operand type does not match comparison operand type",
            s"$location.right",
            comparison.right.span
          )

      case conditional: Conditional[?] =>
        validateExpression(conditional.condition, parameters, s"$location.condition", scope) ++
          validateExpression(conditional.whenTrue, parameters, s"$location.whenTrue", scope) ++
          validateExpression(conditional.whenFalse, parameters, s"$location.whenFalse", scope) ++
          requireSameType(
            conditional.condition.valueType, Bool,
            "conditional expression condition must have CUDA bool type",
            s"$location.condition", conditional.condition.span
          ) ++
          requireSameType(
            conditional.whenTrue.valueType, conditional.valueType,
            "true value type does not match conditional result type",
            s"$location.whenTrue", conditional.whenTrue.span
          ) ++
          requireSameType(
            conditional.whenFalse.valueType, conditional.valueType,
            "false value type does not match conditional result type",
            s"$location.whenFalse", conditional.whenFalse.span
          )

      case math: UnaryMath[?] =>
        validateExpression(math.value, parameters, s"$location.value", scope) ++
          requireSameType(
            math.value.valueType, math.mathType,
            "operand type does not match device math type",
            s"$location.value", math.value.span
          )

      case intrinsic: Intrinsic[?] =>
        intrinsicTypes.get(intrinsic.name) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnknownIntrinsic,
                s"CUDA intrinsic '${intrinsic.name}' is not supported",
                location,
                intrinsic.span
              )
            )
          case Some(expectedType) =>
            requireSameType(
              intrinsic.valueType,
              expectedType,
              s"intrinsic '${intrinsic.name}' has type ${intrinsic.valueType.cudaName}, " +
                s"expected ${expectedType.cudaName}",
              location,
              intrinsic.span
            )

      case conversion: Convert[?, ?] =>
        validateExpression(
          conversion.value,
          parameters,
          s"$location.value",
          scope
        ) ++ validateConversion(conversion, location)

      case accumulation: ToAccumulator[?, ?] =>
        validateExpression(
          accumulation.value,
          parameters,
          s"$location.value",
          scope
        ) ++
          requireSameType(
            accumulation.value.valueType,
            accumulation.rule.inputType,
            "value type does not match accumulation input type",
            s"$location.value",
            accumulation.value.span
          )

      case index: ReductionIndex =>
        if scope.reductionIndexes.contains(index.name) then Vector.empty
        else
          Vector(
            ValidationError(
              ValidationCode.UnboundReductionIndex,
              s"reduction index '${index.name}' is used outside its reduction",
              location,
              index.span
            )
          )

      case index: LoopIndex =>
        if scope.loopIndexes.contains(index.name) then Vector.empty
        else
          Vector(
            ValidationError(
              ValidationCode.UnboundLoopIndex,
              s"loop index '${index.name}' is used outside its loop",
              location,
              index.span
            )
          )

      case reduction: ReduceSum[?, ?] =>
        val indexErrors =
          (if isIdentifier(reduction.index.name) then Vector.empty
           else
             Vector(
               ValidationError(
                 ValidationCode.InvalidReductionIndexName,
                 s"'${reduction.index.name}' is not a valid CUDA reduction index",
                 s"$location.index",
                 reduction.index.span
               )
             )) ++
            (if scope.reductionIndexes.contains(reduction.index.name) ||
                scope.loopIndexes.contains(reduction.index.name) ||
                scope.locals.contains(reduction.index.name) ||
                scope.localArrays.contains(reduction.index.name) ||
                scope.sharedMemory.contains(reduction.index.name)
             then
               Vector(
                 ValidationError(
                   ValidationCode.DuplicateReductionIndex,
                   s"reduction index '${reduction.index.name}' conflicts with an active binding",
                   s"$location.index",
                   reduction.index.span
                 )
               )
             else Vector.empty) ++
            (if parameters.contains(reduction.index.name) then
               Vector(
                 ValidationError(
                   ValidationCode.ReductionIndexConflictsWithParameter,
                   s"reduction index '${reduction.index.name}' conflicts with a parameter",
                   s"$location.index",
                   reduction.index.span
                 )
               )
             else Vector.empty) ++
            (if shadowsConstant(reduction.index.name, scope) then
               Vector(
                 ValidationError(
                   ValidationCode.ConstantNameShadowed,
                   s"reduction index '${reduction.index.name}' shadows a module constant",
                   s"$location.index",
                   reduction.index.span
                 )
               )
             else Vector.empty)

        val rangeErrors =
          validateExpression(
            reduction.from,
            parameters,
            s"$location.from",
            scope
          ) ++
            validateExpression(
              reduction.until,
              parameters,
              s"$location.until",
              scope
            ) ++
            requireSameType(
              reduction.from.valueType,
              I32,
              "reduction lower bound must have CUDA int type",
              s"$location.from",
              reduction.from.span
            ) ++
            requireSameType(
              reduction.until.valueType,
              I32,
              "reduction upper bound must have CUDA int type",
              s"$location.until",
              reduction.until.span
            )

        val accumulatorErrors =
          validateExpression(
            reduction.initial,
            parameters,
            s"$location.initial",
            scope
          ) ++
            requireSameType(
              reduction.initial.valueType,
              reduction.rule.accumulatorType,
              "reduction initial value does not match its accumulator type",
              s"$location.initial",
              reduction.initial.span
            ) ++
            requireSameType(
              reduction.addition,
              reduction.rule.accumulatorType,
              "reduction addition capability does not match its accumulator type",
              location,
              reduction.span
            )

        val valueErrors =
          validateExpression(
            reduction.value,
            parameters,
            s"$location.value",
            scope.copy(
              reductionIndexes =
                scope.reductionIndexes + reduction.index.name
            )
          ) ++
            requireSameType(
              reduction.value.valueType,
              reduction.rule.inputType,
              "reduction value does not match its accumulation input type",
              s"$location.value",
              reduction.value.span
            )

        indexErrors ++ rangeErrors ++ validateLoopStep(reduction.step, location, reduction.span) ++
          accumulatorErrors ++ valueErrors

      case load: Load[?, ?, ?] =>
        validatePlace(
          load.from,
          parameters,
          s"$location.from",
          isWrite = false,
          scope = scope
        )

  private def validateConversion(conversion: Convert[?, ?], location: String): Vector[ValidationError] =
    val from = conversion.value.valueType
    val to = conversion.valueType
    val nearest = Set(RoundingMode.NearestEven)
    val noSaturation = Set(SaturationMode.NoSaturation)
    val policy = (from, to) match
      case (F32, F16) | (F32, BF16) | (I32, F32) | (U32, F32) =>
        Some((RoundingMode.values.toSet, noSaturation))
      case (F32, FP8E4M3) | (F32, FP8E5M2) =>
        Some((nearest, SaturationMode.values.toSet))
      case (F16, F32) | (BF16, F32) | (FP8E4M3, F32) | (FP8E5M2, F32) | (I32, F64) | (U32, F64) =>
        Some((nearest, noSaturation))
      case _ if from == to => Some((nearest, noSaturation))
      case _ => None
    policy match
      case None =>
        Vector(ValidationError(ValidationCode.UnsupportedConversion,
          s"conversion from ${from.cudaName} to ${to.cudaName} is not supported", location, conversion.span))
      case Some((roundingModes, saturationModes)) =>
        val roundingErrors =
          if roundingModes.contains(conversion.rounding) then Vector.empty
          else Vector(ValidationError(ValidationCode.UnsupportedConversionRounding,
            s"conversion from ${from.cudaName} to ${to.cudaName} does not support ${conversion.rounding} rounding",
            location, conversion.span))
        val saturationErrors =
          if saturationModes.contains(conversion.saturation) then Vector.empty
          else Vector(ValidationError(ValidationCode.UnsupportedConversionSaturation,
            s"conversion from ${from.cudaName} to ${to.cudaName} does not support ${conversion.saturation} saturation",
            location, conversion.span))
        roundingErrors ++ saturationErrors

  private def validatePlace(
      place: Place[?, ?, ?],
      parameters: Map[String, KernelParam],
      location: String,
      isWrite: Boolean,
      scope: ValidationScope = ValidationScope()
  ): Vector[ValidationError] =
    place match
      case element: BufferElement[?, ?] =>
        val indexErrors =
          validateExpression(
            element.index,
            parameters,
            s"$location.index",
            scope
          ) ++
            requireSameType(
              element.index.valueType,
              I32,
              "buffer index must have CUDA int type",
              s"$location.index",
              element.index.span
            )

        val declarationErrors = parameters.get(element.bufferName) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnknownBuffer,
                s"buffer '${element.bufferName}' is not declared",
                location,
                element.span
              )
            )

          case Some(_: ScalarParam[?]) =>
            Vector(
              ValidationError(
                ValidationCode.ExpectedBuffer,
                s"parameter '${element.bufferName}' is scalar, not a buffer",
                location,
                element.span
              )
            )

          case Some(buffer: BufferParam[?, ?]) =>
            requireSameType(
              element.valueType,
              buffer.valueType,
              s"buffer element type ${element.valueType.cudaName} does not match " +
                s"parameter type ${buffer.valueType.cudaName}",
              location,
              element.span,
              ValidationCode.BufferTypeMismatch
            ) ++
              (if isWrite && buffer.access == BufferAccess.ReadOnly then
                 Vector(
                   ValidationError(
                     ValidationCode.WriteToReadOnlyBuffer,
                     s"buffer '${element.bufferName}' is read-only",
                     location,
                     element.span
                   )
                 )
               else Vector.empty)

        indexErrors ++ declarationErrors

      case element: ConstantElement[?] =>
        val indexErrors =
          validateArrayIndex(element.index, parameters, s"$location.index", scope)
        val declarationErrors = scope.constants match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.ModuleContextRequired,
                s"constant array '${element.arrayName}' requires module validation",
                location,
                element.span
              )
            )
          case Some(constants) =>
            constants.get(element.arrayName) match
              case None =>
                Vector(
                  ValidationError(
                    ValidationCode.UnknownConstant,
                    s"constant array '${element.arrayName}' is not declared by the module",
                    location,
                    element.span
                  )
                )
              case Some(constant) =>
                val typeErrors = requireSameType(
                  element.valueType,
                  constant.valueType,
                  s"constant element type ${element.valueType.cudaName} does not match " +
                    s"symbol type ${constant.valueType.cudaName}",
                  location,
                  element.span,
                  ValidationCode.ConstantTypeMismatch
                )
                val boundsErrors = validateLiteralArrayIndex(
                  element.index,
                  constant.elementCount,
                  s"constant array '${element.arrayName}'",
                  s"$location.index",
                  ValidationCode.ConstantIndexOutOfBounds
                )

                typeErrors ++ boundsErrors

        indexErrors ++ declarationErrors

      case element: SharedElement[?] =>
        val indexErrors = element.indices.zipWithIndex.flatMap {
          case (index, indexPosition) =>
            validateArrayIndex(
              index,
              parameters,
              s"$location.indices[$indexPosition]",
              scope
            )
        }
        val declarationErrors = scope.sharedMemory.get(element.arrayName) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnknownSharedMemory,
                s"shared array '${element.arrayName}' is not declared by the kernel",
                location,
                element.span
              )
            )
          case Some(memory) =>
            val typeErrors =
              requireSameType(
                element.valueType,
                memory.valueType,
                s"shared element type ${element.valueType.cudaName} does not match " +
                  s"declaration type ${memory.valueType.cudaName}",
                location,
                element.span,
                ValidationCode.SharedMemoryTypeMismatch
              )
            val expectedRank = memory.rankWitness.rank
            val rankErrors =
              if element.indices.size == expectedRank then Vector.empty
              else
                Vector(
                  ValidationError(
                    ValidationCode.SharedMemoryIndexRankMismatch,
                    s"shared array '${element.arrayName}' requires $expectedRank indices, " +
                      s"found ${element.indices.size}",
                    location,
                    element.span
                  )
                )

            val boundsErrors = memory.size match
              case StaticSharedMemory(layout)
                  if element.indices.size == layout.rank =>
                element.indices
                  .zip(layout.logicalDimensions)
                  .zipWithIndex
                  .flatMap { case ((index, upperBound), indexPosition) =>
                    validateLiteralArrayIndex(
                      index,
                      upperBound,
                      s"shared array '${element.arrayName}' index $indexPosition",
                      s"$location.indices[$indexPosition]",
                      ValidationCode.SharedMemoryIndexOutOfBounds
                    )
                  }
              case _ => Vector.empty

            typeErrors ++ rankErrors ++ boundsErrors

        indexErrors ++ declarationErrors

      case element: LocalArrayElement[?] =>
        val indexErrors =
          validateArrayIndex(element.index, parameters, s"$location.index", scope)
        val declarationErrors = scope.localArrays.get(element.arrayName) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnknownLocalArray,
                s"local array '${element.arrayName}' is used before declaration or outside its scope",
                location,
                element.span
              )
            )
          case Some(array) =>
            val typeErrors = requireSameType(
              element.valueType,
              array.valueType,
              s"local array element type ${element.valueType.cudaName} does not match " +
                s"declaration type ${array.valueType.cudaName}",
              location,
              element.span,
              ValidationCode.LocalArrayTypeMismatch
            )
            val boundsErrors = validateLiteralArrayIndex(
              element.index,
              array.elementCount,
              s"local array '${element.arrayName}'",
              s"$location.index",
              ValidationCode.LocalArrayIndexOutOfBounds
            )

            typeErrors ++ boundsErrors

        indexErrors ++ declarationErrors

      case local: LocalVariable[?] =>
        val nameErrors =
          if isIdentifier(local.name) then Vector.empty
          else
            Vector(
              ValidationError(
                ValidationCode.InvalidLocalName,
                s"'${local.name}' is not a valid CUDA local identifier",
                location,
                local.span
              )
            )

        val bindingErrors = scope.locals.get(local.name) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnboundLocal,
                s"local '${local.name}' is used before declaration or outside its scope",
                location,
                local.span
              )
            )
          case Some(declaredType) =>
            requireSameType(
              local.valueType,
              declaredType,
              s"local '${local.name}' has type ${local.valueType.cudaName}, " +
                s"but was declared as ${declaredType.cudaName}",
              location,
              local.span,
              ValidationCode.LocalTypeMismatch
            )

        nameErrors ++ bindingErrors

  private def validateArrayIndex(
      index: Expr[Int],
      parameters: Map[String, KernelParam],
      location: String,
      scope: ValidationScope
  ): Vector[ValidationError] =
    validateExpression(index, parameters, location, scope) ++
      requireSameType(
        index.valueType,
        I32,
        "array index must have CUDA int type",
        location,
        index.span
      )

  private def validateLiteralArrayIndex(
      index: Expr[Int],
      exclusiveUpperBound: Int,
      description: String,
      location: String,
      code: ValidationCode
  ): Vector[ValidationError] =
    index match
      case Literal(value, _, _)
          if exclusiveUpperBound > 0 &&
            (value < 0 || value >= exclusiveUpperBound) =>
        Vector(
          ValidationError(
            code,
            s"$description must be in [0, $exclusiveUpperBound), found $value",
            location,
            index.span
          )
        )
      case _ => Vector.empty

  private def validateLocalName(
      local: LocalVariable[?],
      parameters: Map[String, KernelParam],
      scope: ValidationScope,
      location: String
  ): Vector[ValidationError] =
    val identifierErrors =
      if isIdentifier(local.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidLocalName,
            s"'${local.name}' is not a valid CUDA local identifier",
            s"$location.local",
            local.span
          )
        )

    val duplicateErrors =
      if scope.locals.contains(local.name) ||
          scope.localArrays.contains(local.name)
      then
        Vector(
          ValidationError(
            ValidationCode.DuplicateLocalName,
            s"local '${local.name}' is already declared in an active scope",
            s"$location.local",
            local.span
          )
        )
      else Vector.empty

    val conflictErrors =
      if parameters.contains(local.name) ||
          scope.sharedMemory.contains(local.name) ||
          scope.loopIndexes.contains(local.name) ||
          scope.reductionIndexes.contains(local.name)
      then
        Vector(
          ValidationError(
            ValidationCode.LocalNameConflictsWithBinding,
            s"local '${local.name}' conflicts with an active binding",
            s"$location.local",
            local.span
          )
        )
      else Vector.empty

    val constantShadowErrors =
      if shadowsConstant(local.name, scope) then
        Vector(
          ValidationError(
            ValidationCode.ConstantNameShadowed,
            s"local '${local.name}' shadows a module constant",
            s"$location.local",
            local.span
          )
        )
      else Vector.empty

    identifierErrors ++
      duplicateErrors ++
      conflictErrors ++
      constantShadowErrors

  private def validateLocalArrayDeclaration(
      array: LocalArray[?],
      parameters: Map[String, KernelParam],
      scope: ValidationScope,
      location: String
  ): LocalArrayDeclarationValidation =
    val identifierErrors =
      if isIdentifier(array.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidLocalName,
            s"'${array.name}' is not a valid CUDA local-array identifier",
            s"$location.array",
            array.span
          )
        )

    val duplicateErrors =
      if scope.locals.contains(array.name) ||
          scope.localArrays.contains(array.name)
      then
        Vector(
          ValidationError(
            ValidationCode.DuplicateLocalName,
            s"local binding '${array.name}' is already declared in an active scope",
            s"$location.array",
            array.span
          )
        )
      else Vector.empty

    val conflictErrors =
      if parameters.contains(array.name) ||
          scope.sharedMemory.contains(array.name) ||
          scope.loopIndexes.contains(array.name) ||
          scope.reductionIndexes.contains(array.name)
      then
        Vector(
          ValidationError(
            ValidationCode.LocalNameConflictsWithBinding,
            s"local array '${array.name}' conflicts with an active binding",
            s"$location.array",
            array.span
          )
        )
      else Vector.empty

    val sizeErrors =
      if array.elementCount > 0 then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidLocalArrayElementCount,
            s"local array '${array.name}' must have a positive element count",
            s"$location.array",
            array.span
          )
        )

    val constantShadowErrors =
      if shadowsConstant(array.name, scope) then
        Vector(
          ValidationError(
            ValidationCode.ConstantNameShadowed,
            s"local array '${array.name}' shadows a module constant",
            s"$location.array",
            array.span
          )
        )
      else Vector.empty

    val bindingErrors =
      identifierErrors ++
        duplicateErrors ++
        conflictErrors ++
        constantShadowErrors

    LocalArrayDeclarationValidation(
      errors = bindingErrors ++ sizeErrors,
      bindingErrors = bindingErrors
    )

  private def validateLoopIndexName(
      index: LoopIndex,
      parameters: Map[String, KernelParam],
      scope: ValidationScope,
      location: String
  ): Vector[ValidationError] =
    val identifierErrors =
      if isIdentifier(index.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidLoopIndexName,
            s"'${index.name}' is not a valid CUDA loop index",
            location,
            index.span
          )
        )

    val conflictErrors =
      if parameters.contains(index.name) ||
          scope.locals.contains(index.name) ||
          scope.localArrays.contains(index.name) ||
          scope.sharedMemory.contains(index.name) ||
          scope.loopIndexes.contains(index.name) ||
          scope.reductionIndexes.contains(index.name)
      then
        Vector(
          ValidationError(
            ValidationCode.LoopIndexConflictsWithBinding,
            s"loop index '${index.name}' conflicts with an active binding",
            location,
            index.span
          )
        )
      else Vector.empty

    val constantShadowErrors =
      if shadowsConstant(index.name, scope) then
        Vector(
          ValidationError(
            ValidationCode.ConstantNameShadowed,
            s"loop index '${index.name}' shadows a module constant",
            location,
            index.span
          )
        )
      else Vector.empty

    identifierErrors ++ conflictErrors ++ constantShadowErrors

  private def shadowsConstant(
      name: String,
      scope: ValidationScope
  ): Boolean =
    scope.constants.exists(_.contains(name))

  private def requireSameType(
      actual: CudaType[?],
      expected: CudaType[?],
      message: String,
      location: String,
      span: SourceSpan,
      code: ValidationCode = ValidationCode.ExpressionTypeMismatch
  ): Vector[ValidationError] =
    if actual == expected then Vector.empty
    else Vector(ValidationError(code, message, location, span))

  private def isIdentifier(value: String): Boolean =
    cudaIdentifier.matches(value)
