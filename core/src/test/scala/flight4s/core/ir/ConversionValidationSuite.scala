package flight4s.core.ir

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.{CudaDsl, DslSourcePosition}
import CudaDsl.*
import flight4s.core.types.*

class ConversionValidationSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  private def definition[T](expression: Expr[T]) = kernel("conversion") {
    local("converted", expression)(using expression.valueType, summon[BlockBuilder], summon[DslSourcePosition])
    ()
  }

  test("unsupported type pairs fail structural validation before code generation"):
    val invalid = definition(Convert(literal(3.0), I32, RoundingMode.NearestEven, SaturationMode.NoSaturation))
    assert(!KernelValidator.validate(invalid).isValid)

  test("FP8 conversion cannot silently ignore requested directed rounding"):
    val invalid = definition(Convert(literal(1.1f), FP8E4M3, RoundingMode.TowardZero, SaturationMode.SaturateFinite))
    assert(!KernelValidator.validate(invalid).isValid)
    assert(CudaCodegen.generate(invalid).isLeft)

  test("all scalar type pairs and policy combinations agree with the supported conversion matrix"):
    val sources: Vector[Expr[?]] = Vector(
      literal(false), literal(1), literal(UInt.fromBits(1)), literal(1.0f), literal(1.0),
      literal(Float16.fromBits(0x3c00.toShort)), literal(BFloat16.fromBits(0x3f80.toShort)),
      literal(Float8E4M3.fromBits(0x38.toByte)), literal(Float8E5M2.fromBits(0x3c.toByte))
    )
    val decodePairs = Set[(CudaType[?], CudaType[?])]((F16, F32), (BF16, F32), (FP8E4M3, F32), (FP8E5M2, F32))
    var accepted = 0
    var examined = 0
    for source <- sources; target <- sources.map(_.valueType)
        rounding <- RoundingMode.values; saturation <- SaturationMode.values do
      val from = source.valueType
      val integerSource = from == I32 || from == U32
      val exact = from == target || decodePairs.contains((from, target)) ||
        ((integerSource || from == F32) && target == F64)
      val rounded = (from == F32 && (target == F16 || target == BF16)) ||
        ((integerSource || from == F64) && target == F32)
      val fp8Narrowing = from == F32 && (target == FP8E4M3 || target == FP8E5M2)
      val expected =
        (exact && rounding == RoundingMode.NearestEven && saturation == SaturationMode.NoSaturation) ||
          (rounded && saturation == SaturationMode.NoSaturation) ||
          (fp8Narrowing && rounding == RoundingMode.NearestEven)
      val kernel = definition(Convert(source, target, rounding, saturation))
      val clue = s"${from.cudaName} -> ${target.cudaName}, $rounding, $saturation"
      assertEquals(KernelValidator.validate(kernel).isValid, expected, clue)
      assertEquals(CudaCodegen.generate(kernel).isRight, expected, clue)
      examined += 1
      if expected then accepted += 1
    assertEquals(examined, 648)
    assertEquals(accepted, 40)

  test("conversion diagnostics retain code location and source span"):
    val span = SourceSpan("Convert.scala", 5, 2, 5, 70)
    val expression = Convert(literal(Float16.fromBits(0)), F32,
      RoundingMode.TowardZero, SaturationMode.SaturateFinite, span)
    val errors = KernelValidator.validate(definition(expression)).errors
    assertEquals(errors.map(_.code), Vector(
      ValidationCode.UnsupportedConversionRounding, ValidationCode.UnsupportedConversionSaturation
    ))
    assert(errors.forall(_.span == span))
    assert(errors.forall(_.location.endsWith(".initial")))
    assert(errors.forall(_.message.contains("__half")))
    assert(errors.forall(_.message.contains("float")))

  test("unsupported type pairs do not cascade into policy errors but child errors remain"):
    val expression = Convert(input[Double]("missing")(literal(0)).read, I32,
      RoundingMode.TowardZero, SaturationMode.SaturateFinite)
    assertEquals(KernelValidator.validate(definition(expression)).errors.map(_.code),
      Vector(ValidationCode.UnknownBuffer, ValidationCode.UnsupportedConversion))

  test("unselected invalid conversions fail before normalization can discard them"):
    val bad = Convert(literal(1.0f), FP8E5M2, RoundingMode.TowardPositive, SaturationMode.NoSaturation)
    val expression = choose(literal(false))(bad)(literal(Float8E5M2.fromBits(0)))
    assertEquals(KernelValidator.validate(definition(expression)).errors.map(_.code),
      Vector(ValidationCode.UnsupportedConversionRounding))
    assert(CudaCodegen.generate(definition(expression)).isLeft)

  test("public low-precision helpers preserve their existing generated conversion calls"):
    val half = CudaCodegen.generate(definition(convert.f32ToF16(literal(1.0f), RoundingMode.TowardNegative)))
      .toOption.get.cudaSource
    val bfloat = CudaCodegen.generate(definition(convert.f32ToBF16(literal(1.0f), RoundingMode.TowardPositive)))
      .toOption.get.cudaSource
    val fp8 = CudaCodegen.generate(definition(convert.f32ToFP8E4M3(literal(1.0f))))
      .toOption.get.cudaSource
    assert(half.contains("__float2half_rd(0x1.0p0f)"))
    assert(bfloat.contains("__float2bfloat16_ru(0x1.0p0f)"))
    assert(fp8.contains("__NV_SATFINITE, __NV_E4M3"))

  test("half conversion cannot silently ignore finite saturation"):
    val invalid = definition(Convert(literal(Float.PositiveInfinity), F16,
      RoundingMode.NearestEven, SaturationMode.SaturateFinite))
    assert(!KernelValidator.validate(invalid).isValid)
    assert(CudaCodegen.generate(invalid).isLeft)
