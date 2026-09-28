package flight4s.core.codegen

import scala.collection.mutable.ArrayBuffer

import flight4s.core.codegen.CodegenError.*
import flight4s.core.ir.*
import flight4s.core.types.*

object CudaCodegen:
  val ArtifactVersion: Int = 23

  def generate[Args <: Tuple](
      kernel: Kernel[Args],
      compilerOptions: CompilerOptions = CompilerOptions()
  ): Either[CodegenError, GeneratedKernel[Args]] =
    generateModule(
      CudaModuleIR(Vector.empty, Vector(kernel.ir)),
      compilerOptions
    ).map { generated =>
      GeneratedKernel(
        name = kernel.name,
        signature = kernel.signature,
        cudaSource = generated.cudaSource,
        sourceMap = generated.sourceMap,
        compilerOptions = generated.compilerOptions,
        declarationLine = generated.kernels.head.declarationLine,
        launchRequirements = generated.kernels.head.launchRequirements
      )
    }

  def generateModule(
      module: CudaModuleIR,
      compilerOptions: CompilerOptions = CompilerOptions()
  ): Either[CodegenError, GeneratedCudaModule] =
    val validation = ModuleValidator.validate(module)
    if !validation.isValid then Left(ValidationFailed(validation.errors))
    else ModuleEmitter(IrNormalizer.module(module), compilerOptions).emit()

  private final class SourceWriter:
    private val lines = ArrayBuffer.empty[String]
    private val mappings = ArrayBuffer.empty[SourceMapEntry]

    def line(
        text: String = "",
        span: SourceSpan = SourceSpan.Unknown
    ): Int =
      lines += text
      val generatedLine = lines.size
      if span != SourceSpan.Unknown then
        mappings += SourceMapEntry(generatedLine, span)
      generatedLine

    def source: String =
      if lines.isEmpty then "" else lines.mkString("", "\n", "\n")

    def sourceMap: SourceMap =
      SourceMap(mappings.toVector)

  private final class FreshNames(occupied: Set[String]):
    private var nextId = 0

    def accumulator(): String =
      next("accumulator")

    def value(): String =
      next("value")

    private def next(label: String): String =
      var candidate = s"flight4s_${label}_$nextId"
      nextId += 1
      while occupied.contains(candidate) do
        candidate = s"flight4s_${label}_$nextId"
        nextId += 1
      candidate

  private final case class ModuleEmitter(
      module: CudaModuleIR,
      compilerOptions: CompilerOptions
  ):
    private val writer = SourceWriter()

    def emit(): Either[CodegenError, GeneratedCudaModule] =
      emitHeaders()
      emitConstants()

      val generatedKernels = ArrayBuffer.empty[(KernelIR[?], Int)]
      val kernels = module.kernels.iterator
      var error: Option[CodegenError] = None

      while kernels.hasNext && error.isEmpty do
        val kernel = kernels.next()
        KernelEmitter(
          kernel,
          writer,
          module.constants.map(_.name).toSet
        ).emit() match
          case Right(declarationLine) =>
            generatedKernels += ((kernel, declarationLine))
            if kernels.hasNext then writer.line()
          case Left(codegenError) =>
            error = Some(codegenError)

      error match
        case Some(codegenError) =>
          Left(codegenError)
        case None =>
          val source = writer.source
          val sourceMap = writer.sourceMap
          Right(
            GeneratedCudaModule(
              cudaSource = source,
              sourceMap = sourceMap,
              compilerOptions = compilerOptions,
              kernels = generatedKernels.toVector.map {
                case (kernel, declarationLine) =>
                  generatedKernel(
                    kernel,
                    source,
                    sourceMap,
                    declarationLine
                  )
              }
            )
          )

    private def generatedKernel[Args <: Tuple](
        kernel: KernelIR[Args],
        source: String,
        sourceMap: SourceMap,
        declarationLine: Int
    ): GeneratedKernel[Args] =
      GeneratedKernel(
        name = kernel.name,
        signature = kernel.signature,
        cudaSource = source,
        sourceMap = sourceMap,
        compilerOptions = compilerOptions,
        declarationLine = declarationLine,
        launchRequirements = KernelLaunchRequirements(
          dynamicSharedMemory =
            kernel.sharedMemory.collectFirst {
              case SharedArray(
                    _,
                    valueType,
                    DynamicSharedMemory,
                    _
                  ) =>
                DynamicSharedMemoryRequirement(
                  elementSizeBytes = valueType.sizeBytes,
                  elementAlignmentBytes = valueType.alignmentBytes
                )
            },
          requiredBlock = kernel.requiredBlock
        )
      )

    private def emitHeaders(): Unit =
      val headers = collectTypes(module)
        .flatMap(_.requiredHeaders)
        .toVector
        .sorted

      headers.foreach(header => writer.line(s"#include <$header>"))
      if headers.nonEmpty then writer.line()

    private def emitConstants(): Unit =
      module.constants.zipWithIndex.foreach { case (constant, index) =>
        writer.line(
          s"""extern "C" __constant__ ${constant.valueType.cudaName} ${constant.name}[${constant.elementCount}];""",
          constant.span
        )
        if index == module.constants.size - 1 && module.kernels.nonEmpty then
          writer.line()
      }

  private final case class KernelEmitter(
      kernel: KernelIR[?],
      writer: SourceWriter,
      moduleSymbols: Set[String]
  ):
    private val freshNames =
      FreshNames(collectIdentifiers(kernel) ++ moduleSymbols)

    def emit(): Either[CodegenError, Int] =
      val parameters = kernel.params.map(emitParameter).mkString(", ")
      val declarationLine = writer.line(
        s"""extern "C" __global__ void ${kernel.name}($parameters) {"""
      )

      kernel.sharedMemory.foreach(emitSharedMemory)
      if kernel.sharedMemory.nonEmpty && kernel.body.statements.nonEmpty then
        writer.line()

      emitBlock(kernel.body, indentation = 1).map { _ =>
        writer.line("}")
        declarationLine
      }

    private def emitParameter(parameter: KernelParam): String =
      parameter match
        case scalar: ScalarParam[?] =>
          s"${scalar.valueType.cudaName} ${scalar.name}"
        case buffer: BufferParam[?, ?] =>
          val constQualifier =
            if buffer.access == BufferAccess.ReadOnly then "const " else ""
          s"$constQualifier${buffer.valueType.cudaName}* ${buffer.name}"

    private def emitSharedMemory(shared: SharedArray[?, ?]): Unit =
      val indentation = indent(1)
      shared.size match
        case static: StaticSharedMemory[?] =>
          val dimensions =
            static.layout.physicalDimensions.map(size => s"[$size]").mkString
          writer.line(
            s"${indentation}__shared__ ${shared.valueType.cudaName} ${shared.name}$dimensions;",
            shared.span
          )
        case DynamicSharedMemory =>
          writer.line(
            s"$indentation" +
              s"extern __shared__ __align__(${shared.valueType.alignmentBytes}) " +
              s"${shared.valueType.cudaName} ${shared.name}[];",
            shared.span
          )

    private def emitBlock(
        block: Block,
        indentation: Int
    ): Either[CodegenError, Unit] =
      block.statements.foldLeft[Either[CodegenError, Unit]](Right(())) {
        case (result, statement) =>
          result.flatMap(_ => emitStatement(statement, indentation))
      }

    private def emitStatement(
        statement: Stmt,
        indentation: Int
    ): Either[CodegenError, Unit] =
      val prefix = indent(indentation)
      statement match
        case declaration: LocalDeclaration[?] =>
          emitExpression(declaration.initial).map { initial =>
            writer.line(
              s"$prefix${declaration.local.valueType.cudaName} " +
                s"${declaration.local.name} = $initial;",
              declaration.span
            )
          }

        case declaration: LocalArrayDeclaration[?] =>
          writer.line(
            s"$prefix${declaration.array.valueType.cudaName} " +
              s"${declaration.array.name}[${declaration.array.elementCount}];",
            declaration.span
          )
          Right(())

        case store: Store[?, ?] =>
          for
            target <- emitPlace(store.to)
            value <- emitExpression(store.value)
          yield writer.line(s"$prefix$target = $value;", store.span)

        case shuffle: WarpShuffle[?, ?] =>
          for
            mask <- emitExpression(shuffle.mask)
            value <- emitExpression(shuffle.value)
            selector <- emitExpression(shuffle.selector)
          yield writer.line(
            s"$prefix${shuffle.local.valueType.cudaName} ${shuffle.local.name} = ::${shuffle.operator.cudaName}($mask, $value, $selector, ${shuffle.width});",
            shuffle.span
          )

        case vote: WarpVote[?] =>
          for
            mask <- emitExpression(vote.mask)
            predicate <- emitExpression(vote.predicate)
          yield writer.line(
            s"$prefix${vote.local.valueType.cudaName} ${vote.local.name} = ::${vote.operator.cudaName}($mask, $predicate);",
            vote.span
          )

        case atomic: AtomicFetchAdd[?, ?] =>
          for
            target <- emitPlace(atomic.target)
            value <- emitExpression(atomic.value)
          yield writer.line(
            s"$prefix${atomic.local.valueType.cudaName} ${atomic.local.name} = ::atomicAdd(&$target, $value);",
            atomic.span
          )

        case atomic: AtomicAdd[?, ?] =>
          for
            target <- emitPlace(atomic.target)
            value <- emitExpression(atomic.value)
          yield writer.line(s"${prefix}::atomicAdd(&$target, $value);", atomic.span)

        case accumulation: Accumulate[?] =>
          emitExpression(accumulation.value).map { value =>
            writer.line(
              s"$prefix${accumulation.target.name} += $value;",
              accumulation.span
            )
          }

        case branch: IfThen =>
          emitExpression(branch.condition).flatMap { condition =>
            writer.line(s"$prefix" + s"if ($condition) {", branch.span)
            emitBlock(branch.thenBlock, indentation + 1).flatMap { _ =>
              branch.elseBlock match
                case Some(elseBlock) =>
                  writer.line(s"$prefix} else {")
                  emitBlock(elseBlock, indentation + 1).map { _ =>
                    writer.line(s"$prefix}")
                  }
                case None =>
                  writer.line(s"$prefix}")
                  Right(())
            }
          }

        case scoped: ScopedBlock =>
          writer.line(s"$prefix{", scoped.span)
          emitBlock(scoped.body, indentation + 1).map { _ =>
            writer.line(s"$prefix}")
          }

        case loop: ForLoop =>
          for
            from <- emitExpression(loop.from)
            until <- emitExpression(loop.until)
            _ =
              val (header, binding) = loopOpening(loop.index.name, from, until, loop.step)
              writer.line(s"$prefix$header", loop.span)
              binding.foreach(line => writer.line(s"${indent(indentation + 1)}$line", loop.index.span))
            _ <- emitBlock(loop.body, indentation + 1)
          yield writer.line(s"$prefix}")

        case barrier: Barrier =>
          writer.line(s"${prefix}__syncthreads();", barrier.span)
          Right(())

        case barrier: WarpBarrier =>
          emitExpression(barrier.mask).map { mask =>
            writer.line(s"${prefix}::__syncwarp($mask);", barrier.span)
          }

    private def emitExpression(
        expression: Expr[?]
    ): Either[CodegenError, String] =
      expression match
        case literal: Literal[?] =>
          emitLiteral(literal)

        case scalar: ScalarParam[?] =>
          Right(scalar.name)

        case binary: Binary[?] =>
          for
            left <- emitExpression(binary.left)
            right <- emitExpression(binary.right)
          yield s"($left ${binary.operator.cudaToken} $right)"

        case shift: UnsignedShift =>
          for
            value <- emitExpression(shift.value)
            distance <- emitExpression(shift.distance)
          yield s"($value ${shift.operator.cudaToken} ($distance & 31))"

        case shift: SignedShift => emitSignedShift(shift)

        case count: PopulationCount[?] =>
          emitExpression(count.value).map(value => s"::__popc(static_cast<unsigned int>($value))")

        case comparison: Compare[?] =>
          for
            left <- emitExpression(comparison.left)
            right <- emitExpression(comparison.right)
          yield s"($left ${comparison.operator.cudaToken} $right)"

        case conditional: Conditional[?] =>
          for
            condition <- emitExpression(conditional.condition)
            whenTrue <- emitExpression(conditional.whenTrue)
            whenFalse <- emitExpression(conditional.whenFalse)
          yield s"($condition ? $whenTrue : $whenFalse)"

        case math: UnaryMath[?] =>
          val suffix = math.mathType match
            case F32 => "f"
            case F64 => ""
          emitExpression(math.value).map(value => s"::${math.operator.cudaName}$suffix($value)")

        case intrinsic: Intrinsic[?] =>
          Right(intrinsic.name)

        case conversion: Convert[?, ?] =>
          emitExpression(conversion.value).flatMap { value =>
            emitConversion(conversion, value)
          }

        case accumulation: ToAccumulator[?, ?] =>
          emitExpression(accumulation.value).flatMap { value =>
            emitAccumulatorConversion(
              accumulation.value.valueType,
              accumulation.valueType,
              value,
              accumulation.span
            )
          }

        case index: ReductionIndex =>
          Right(index.name)

        case index: LoopIndex =>
          Right(index.name)

        case reduction: ReduceSum[?, ?] =>
          emitReduction(reduction)

        case load: Load[?, ?, ?] =>
          emitPlace(load.from)

    private def emitSignedShift(shift: SignedShift): Either[CodegenError, String] =
      for
        value <- emitExpression(shift.value)
        distance <- emitExpression(shift.distance)
      yield
        val bits = freshNames.value()
        val count = freshNames.value()
        val result = freshNames.value()
        val shifted = shift.operator match
          case SignedShiftOperator.Left => s"($bits << $count)"
          case SignedShiftOperator.LogicalRight => s"($bits >> $count)"
          case SignedShiftOperator.ArithmeticRight =>
            s"(($bits & 0x80000000u) != 0u ? ~(~$bits >> $count) : ($bits >> $count))"
        // Convert back only after bounding each signed cast to 0..Int.MaxValue.
        s"([&]() { const unsigned int $bits = static_cast<unsigned int>($value); " +
          s"const unsigned int $count = static_cast<unsigned int>($distance) & 31u; " +
          s"const unsigned int $result = $shifted; " +
          s"return $result <= 0x7fffffffu ? static_cast<int>($result) : (-1 - static_cast<int>(~$result)); }())"

    private def emitPlace(
        place: Place[?, ?, ?]
    ): Either[CodegenError, String] =
      place match
        case buffer: BufferElement[?, ?] =>
          emitExpression(buffer.index).map(index =>
            s"${buffer.bufferName}[$index]"
          )

        case constant: ConstantElement[?] =>
          emitExpression(constant.index).map(index =>
            s"${constant.arrayName}[$index]"
          )

        case shared: SharedElement[?] =>
          sequence(shared.indices.map(emitExpression)).map { indices =>
            shared.arrayName + indices.map(index => s"[$index]").mkString
          }

        case local: LocalVariable[?] =>
          Right(local.name)

        case local: LocalArrayElement[?] =>
          emitExpression(local.index).map(index =>
            s"${local.arrayName}[$index]"
          )

    private def loopOpening(index: String, from: String, until: String, step: Int): (String, Option[String]) =
      if step == 1 then
        (s"for (int $index = $from; $index < $until; ++$index) {", None)
      else
        // The final positive increment may exceed Int.MaxValue.
        val induction = freshNames.value()
        (s"for (long long $induction = $from; $induction < $until; $induction += ${step}LL) {",
          Some(s"const int $index = static_cast<int>($induction);"))

    private def emitReduction(
        reduction: ReduceSum[?, ?]
    ): Either[CodegenError, String] =
      for
        from <- emitExpression(reduction.from)
        until <- emitExpression(reduction.until)
        initial <- emitExpression(reduction.initial)
        value <- emitExpression(reduction.value)
        accumulatedValue <- emitAccumulatorConversion(
          reduction.value.valueType,
          reduction.valueType,
          value,
          reduction.value.span
        )
      yield
        val lowering = ReductionLoweringStrategy.select(reduction.policy)
        val accumulator = freshNames.accumulator()
        val accumulatorType = reduction.valueType.cudaName
        val (header, binding) = loopOpening(reduction.index.name, from, until, reduction.step)
        s"([&]() { /* flight4s reduction: ${lowering.cudaMarker} */ " +
          s"$accumulatorType $accumulator = $initial; " +
          s"$header " + binding.fold("")(_ + " ") +
          s"$accumulator += $accumulatedValue; } return $accumulator; }())"

    private def emitConversion(
        conversion: Convert[?, ?],
        value: String
    ): Either[CodegenError, String] =
      (conversion.value.valueType, conversion.valueType) match
        case (I32, F32) =>
          Right(s"::__int2float_${roundingSuffix(conversion.rounding)}($value)")
        case (U32, F32) =>
          Right(s"::__uint2float_${roundingSuffix(conversion.rounding)}($value)")
        case (I32, F64) =>
          Right(s"::__int2double_rn($value)")
        case (U32, F64) =>
          Right(s"::__uint2double_rn($value)")
        case (F32, F16) =>
          Right(s"${halfConversion(conversion.rounding)}($value)")
        case (F16, F32) =>
          Right(s"__half2float($value)")
        case (F32, BF16) =>
          Right(s"${bfloat16Conversion(conversion.rounding)}($value)")
        case (BF16, F32) =>
          Right(s"__bfloat162float($value)")
        case (F32, FP8E4M3) =>
          Right(
            emitFloatToFp8(
              value,
              "__nv_fp8_e4m3",
              "__NV_E4M3",
              conversion.saturation
            )
          )
        case (FP8E4M3, F32) =>
          Right(s"static_cast<float>($value)")
        case (F32, FP8E5M2) =>
          Right(
            emitFloatToFp8(
              value,
              "__nv_fp8_e5m2",
              "__NV_E5M2",
              conversion.saturation
            )
          )
        case (FP8E5M2, F32) =>
          Right(s"static_cast<float>($value)")
        case (fromType, toType) if fromType == toType =>
          Right(value)
        case (fromType, toType) =>
          Left(
            UnsupportedConversion(
              fromType.cudaName,
              toType.cudaName,
              conversion.span
            )
          )

    private def emitAccumulatorConversion(
        fromType: CudaType[?],
        toType: CudaType[?],
        value: String,
        span: SourceSpan
    ): Either[CodegenError, String] =
      (fromType, toType) match
        case (from, to) if from == to =>
          Right(value)
        case (F16, F32) =>
          Right(s"__half2float($value)")
        case (BF16, F32) =>
          Right(s"__bfloat162float($value)")
        case (FP8E4M3, F32) | (FP8E5M2, F32) =>
          Right(s"static_cast<float>($value)")
        case _ =>
          Left(
            UnsupportedConversion(
              fromType.cudaName,
              toType.cudaName,
              span
            )
          )

    private def emitLiteral(
        literal: Literal[?]
    ): Either[CodegenError, String] =
      (literal.valueType, literal.value) match
        case (Bool, value: Boolean) =>
          Right(if value then "true" else "false")
        case (I32, value: Int) =>
          Right(
            if value == Int.MinValue then "(-2147483647 - 1)"
            else value.toString
          )
        case (U32, value: UInt) =>
          Right(f"0x${value.toIntBits}%08xu")
        case (F16, value: Float16) =>
          val bits = value.toShortBits & 0xffff
          Right(f"__half(__half_raw{0x$bits%04xu})")
        case (BF16, value: BFloat16) =>
          val bits = value.toShortBits & 0xffff
          Right(f"__nv_bfloat16(__nv_bfloat16_raw{0x$bits%04xu})")
        case (F32, value: Float) =>
          Right(floatLiteral(value))
        case (F64, value: Double) =>
          Right(doubleLiteral(value))
        case (FP8E4M3, value: Float8E4M3) =>
          Right(fp8Literal("__nv_fp8_e4m3", value.toByteBits & 0xff))
        case (FP8E5M2, value: Float8E5M2) =>
          Right(fp8Literal("__nv_fp8_e5m2", value.toByteBits & 0xff))
        case _ =>
          Left(
            UnsupportedLiteral(
              literal.valueType.cudaName,
              literal.value.getClass.getName,
              literal.span
            )
          )

    private def roundingSuffix(rounding: RoundingMode): String =
      rounding match
        case RoundingMode.NearestEven => "rn"
        case RoundingMode.TowardZero => "rz"
        case RoundingMode.TowardPositive => "ru"
        case RoundingMode.TowardNegative => "rd"

    private def halfConversion(rounding: RoundingMode): String =
      rounding match
        case RoundingMode.NearestEven => "__float2half_rn"
        case RoundingMode.TowardZero => "__float2half_rz"
        case RoundingMode.TowardPositive => "__float2half_ru"
        case RoundingMode.TowardNegative => "__float2half_rd"

    private def bfloat16Conversion(rounding: RoundingMode): String =
      rounding match
        case RoundingMode.NearestEven => "__float2bfloat16_rn"
        case RoundingMode.TowardZero => "__float2bfloat16_rz"
        case RoundingMode.TowardPositive => "__float2bfloat16_ru"
        case RoundingMode.TowardNegative => "__float2bfloat16_rd"

    private def emitFloatToFp8(
        value: String,
        cudaType: String,
        interpretation: String,
        saturation: SaturationMode
    ): String =
      val saturationToken = saturation match
        case SaturationMode.NoSaturation => "__NV_NOSAT"
        case SaturationMode.SaturateFinite => "__NV_SATFINITE"
      val temporary = freshNames.value()
      s"([&]() { $cudaType $temporary; " +
        s"$temporary.__x = __nv_cvt_float_to_fp8(" +
        s"$value, $saturationToken, $interpretation); " +
        s"return $temporary; }())"

    private def fp8Literal(cudaType: String, bits: Int): String =
      val temporary = freshNames.value()
      f"([&]() { $cudaType $temporary; " +
        f"$temporary.__x = 0x$bits%02xu; return $temporary; }())"

  private def sequence(
      values: Vector[Either[CodegenError, String]]
  ): Either[CodegenError, Vector[String]] =
    values.foldLeft[Either[CodegenError, Vector[String]]](Right(Vector.empty)) {
      case (result, value) =>
        for
          accumulated <- result
          next <- value
        yield accumulated :+ next
    }

  private def collectTypes(module: CudaModuleIR): Set[CudaType[?]] =
    module.constants.map(_.valueType).toSet ++
      module.kernels.flatMap(collectTypes).toSet

  private def collectTypes(kernel: KernelIR[?]): Vector[CudaType[?]] =
    kernel.params.map(_.valueType) ++
      kernel.sharedMemory.map(_.valueType) ++
      collectTypes(kernel.body)

  private def collectTypes(block: Block): Vector[CudaType[?]] =
    block.statements.flatMap {
      case declaration: LocalDeclaration[?] =>
        declaration.local.valueType +: collectTypes(declaration.initial)
      case declaration: LocalArrayDeclaration[?] =>
        Vector(declaration.array.valueType)
      case vote: WarpVote[?] =>
        vote.local.valueType +: (collectTypes(vote.mask) ++ collectTypes(vote.predicate))
      case shuffle: WarpShuffle[?, ?] =>
        shuffle.local.valueType +: (collectTypes(shuffle.mask) ++ collectTypes(shuffle.value) ++ collectTypes(shuffle.selector))
      case store: Store[?, ?] =>
        collectTypes(store.to) ++ collectTypes(store.value)
      case atomic: AtomicAdd[?, ?] =>
        collectTypes(atomic.target) ++ collectTypes(atomic.value)
      case atomic: AtomicFetchAdd[?, ?] =>
        atomic.local.valueType +: (collectTypes(atomic.target) ++ collectTypes(atomic.value))
      case accumulation: Accumulate[?] =>
        accumulation.target.valueType +:
          collectTypes(accumulation.value)
      case branch: IfThen =>
        collectTypes(branch.condition) ++
          collectTypes(branch.thenBlock) ++
          branch.elseBlock.toVector.flatMap(collectTypes)
      case scoped: ScopedBlock =>
        collectTypes(scoped.body)
      case loop: ForLoop =>
        collectTypes(loop.from) ++
          collectTypes(loop.until) ++
          collectTypes(loop.body)
      case _: Barrier =>
        Vector.empty
      case barrier: WarpBarrier => collectTypes(barrier.mask)
    }

  private def collectTypes(expression: Expr[?]): Vector[CudaType[?]] =
    expression.valueType +: (expression match
      case _: Literal[?] | _: ScalarParam[?] | _: Intrinsic[?] |
          _: ReductionIndex | _: LoopIndex =>
        Vector.empty
      case binary: Binary[?] =>
        collectTypes(binary.left) ++ collectTypes(binary.right)
      case shift: UnsignedShift => collectTypes(shift.value) ++ collectTypes(shift.distance)
      case shift: SignedShift => collectTypes(shift.value) ++ collectTypes(shift.distance)
      case count: PopulationCount[?] => collectTypes(count.value)
      case comparison: Compare[?] =>
        collectTypes(comparison.left) ++ collectTypes(comparison.right)
      case conditional: Conditional[?] =>
        collectTypes(conditional.condition) ++
          collectTypes(conditional.whenTrue) ++ collectTypes(conditional.whenFalse)
      case math: UnaryMath[?] => collectTypes(math.value)
      case conversion: Convert[?, ?] =>
        collectTypes(conversion.value)
      case accumulation: ToAccumulator[?, ?] =>
        collectTypes(accumulation.value)
      case reduction: ReduceSum[?, ?] =>
        collectTypes(reduction.from) ++
          collectTypes(reduction.until) ++
          collectTypes(reduction.initial) ++
          collectTypes(reduction.value)
      case load: Load[?, ?, ?] =>
        collectTypes(load.from))

  private def collectTypes(place: Place[?, ?, ?]): Vector[CudaType[?]] =
    place.valueType +: (place match
      case buffer: BufferElement[?, ?] =>
        collectTypes(buffer.index)
      case constant: ConstantElement[?] =>
        collectTypes(constant.index)
      case shared: SharedElement[?] =>
        shared.indices.flatMap(collectTypes)
      case _: LocalVariable[?] =>
        Vector.empty
      case local: LocalArrayElement[?] =>
        collectTypes(local.index))

  private def collectIdentifiers(kernel: KernelIR[?]): Set[String] =
    kernel.params.map(_.name).toSet ++
      kernel.sharedMemory.map(_.name) ++
      collectIdentifiers(kernel.body)

  private def collectIdentifiers(block: Block): Set[String] =
    block.statements.flatMap {
      case declaration: LocalDeclaration[?] =>
        collectIdentifiers(declaration.initial) + declaration.local.name
      case declaration: LocalArrayDeclaration[?] =>
        Set(declaration.array.name)
      case vote: WarpVote[?] =>
        collectIdentifiers(vote.mask) ++ collectIdentifiers(vote.predicate) + vote.local.name
      case shuffle: WarpShuffle[?, ?] =>
        collectIdentifiers(shuffle.mask) ++ collectIdentifiers(shuffle.value) ++ collectIdentifiers(shuffle.selector) + shuffle.local.name
      case store: Store[?, ?] =>
        collectIdentifiers(store.to) ++ collectIdentifiers(store.value)
      case atomic: AtomicAdd[?, ?] =>
        collectIdentifiers(atomic.target) ++ collectIdentifiers(atomic.value)
      case atomic: AtomicFetchAdd[?, ?] =>
        collectIdentifiers(atomic.target) ++ collectIdentifiers(atomic.value) + atomic.local.name
      case accumulation: Accumulate[?] =>
        collectIdentifiers(accumulation.value) + accumulation.target.name
      case branch: IfThen =>
        collectIdentifiers(branch.condition) ++
          collectIdentifiers(branch.thenBlock) ++
          branch.elseBlock.toSet.flatMap(collectIdentifiers)
      case scoped: ScopedBlock =>
        collectIdentifiers(scoped.body)
      case loop: ForLoop =>
        collectIdentifiers(loop.from) ++
          collectIdentifiers(loop.until) ++
          collectIdentifiers(loop.body) +
          loop.index.name
      case _: Barrier =>
        Set.empty[String]
      case barrier: WarpBarrier => collectIdentifiers(barrier.mask)
    }.toSet

  private def collectIdentifiers(expression: Expr[?]): Set[String] =
    expression match
      case _: Literal[?] | _: Intrinsic[?] =>
        Set.empty
      case scalar: ScalarParam[?] =>
        Set(scalar.name)
      case binary: Binary[?] =>
        collectIdentifiers(binary.left) ++ collectIdentifiers(binary.right)
      case shift: UnsignedShift => collectIdentifiers(shift.value) ++ collectIdentifiers(shift.distance)
      case shift: SignedShift => collectIdentifiers(shift.value) ++ collectIdentifiers(shift.distance)
      case count: PopulationCount[?] => collectIdentifiers(count.value)
      case comparison: Compare[?] =>
        collectIdentifiers(comparison.left) ++
          collectIdentifiers(comparison.right)
      case conditional: Conditional[?] =>
        collectIdentifiers(conditional.condition) ++
          collectIdentifiers(conditional.whenTrue) ++ collectIdentifiers(conditional.whenFalse)
      case math: UnaryMath[?] => collectIdentifiers(math.value)
      case conversion: Convert[?, ?] =>
        collectIdentifiers(conversion.value)
      case accumulation: ToAccumulator[?, ?] =>
        collectIdentifiers(accumulation.value)
      case index: ReductionIndex =>
        Set(index.name)
      case index: LoopIndex =>
        Set(index.name)
      case reduction: ReduceSum[?, ?] =>
        collectIdentifiers(reduction.from) ++
          collectIdentifiers(reduction.until) ++
          collectIdentifiers(reduction.initial) ++
          collectIdentifiers(reduction.value) +
          reduction.index.name
      case load: Load[?, ?, ?] =>
        collectIdentifiers(load.from)

  private def collectIdentifiers(place: Place[?, ?, ?]): Set[String] =
    place match
      case buffer: BufferElement[?, ?] =>
        collectIdentifiers(buffer.index) + buffer.bufferName
      case constant: ConstantElement[?] =>
        collectIdentifiers(constant.index) + constant.arrayName
      case shared: SharedElement[?] =>
        shared.indices.flatMap(collectIdentifiers).toSet + shared.arrayName
      case local: LocalVariable[?] =>
        Set(local.name)
      case local: LocalArrayElement[?] =>
        collectIdentifiers(local.index) + local.arrayName

  private def floatLiteral(value: Float): String =
    if java.lang.Float.isFinite(value) then
      java.lang.Float.toHexString(value) + "f"
    else
      val bits = java.lang.Float.floatToRawIntBits(value)
      f"__int_as_float(0x$bits%08xu)"

  private def doubleLiteral(value: Double): String =
    if java.lang.Double.isFinite(value) then
      java.lang.Double.toHexString(value)
    else
      val bits = java.lang.Double.doubleToRawLongBits(value)
      f"__longlong_as_double(0x$bits%016xull)"

  private def indent(level: Int): String =
    "  " * level
