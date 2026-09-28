package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class FloatNarrowingSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("Double expressions have an explicit Float narrowing helper"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.{Expr, RoundingMode}
      val source: Expr[Double] = literal(1.0)
      val nearest: Expr[Float] = convert.f64ToF32(source)
      val downward: Expr[Float] = convert.f64ToF32(source, RoundingMode.TowardNegative)
    """), Nil)

  test("Double narrowing accepts every rounding mode and emits its CUDA intrinsic"):
    val suffixes = Vector(
      RoundingMode.NearestEven -> "rn", RoundingMode.TowardZero -> "rz",
      RoundingMode.TowardNegative -> "rd", RoundingMode.TowardPositive -> "ru")
    for (rounding, suffix) <- suffixes do
      val source = value[Double]("source")
      val out = output[Float]("out")
      val definition = kernel("narrow", params(source, out)) { _ =>
        out(literal(0)) := Convert(source, F32, rounding, SaturationMode.NoSaturation)
      }
      assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
      val generated = CudaCodegen.generate(definition).toOption.get
      assert(generated.cudaSource.contains(s"out[0] = ::__double2float_$suffix(source);"))
      assertEquals(generated.compilerOptions.additionalNvrtcOptions, Vector.empty)

  test("narrowing rejects saturation without rejecting supported rounding"):
    val span = SourceSpan("Narrow.scala", 8, 2, 8, 65)
    for rounding <- RoundingMode.values do
      val definition = kernel("badNarrow") {
        local("narrow", Convert(literal(1.0), F32, rounding, SaturationMode.SaturateFinite, span))
        ()
      }
      val errors = KernelValidator.validate(definition).errors
      assertEquals(errors.map(_.code), Vector(ValidationCode.UnsupportedConversionSaturation))
      assert(errors.forall(_.span == span))
      assert(errors.forall(_.location.endsWith(".initial")))
      assert(CudaCodegen.generate(definition).isLeft)

  test("narrowing retains spans and analysis without JVM rounding"):
    val span = SourceSpan("Narrow.scala", 14, 2, 14, 60)
    for rounding <- RoundingMode.values do
      val narrowed = convert.f64ToF32(literal(1.0 + math.pow(2.0, -24)), rounding)(using DslSourcePosition(span))
      assertEquals(narrowed, Convert(literal(1.0 + math.pow(2.0, -24)), F32,
        rounding, SaturationMode.NoSaturation, span))
      assertEquals(IrNormalizer.expression(narrowed), narrowed)
    val source = input[Double]("source")
    val narrowedLoad = convert.f64ToF32(source(literal(0)).read)
    assertEquals(EffectAnalysis.expression(narrowedLoad).readSpaces, Set(EffectMemorySpace.Global))
    assertEquals(UniformityAnalysis.expression(narrowedLoad), Uniformity.Varying)
    assertEquals(UniformityAnalysis.expression(convert.f64ToF32(literal(1.0))), Uniformity.GridUniform)

  test("Double tuple state can explicitly produce Float output"):
    val values = input[Float]("values")
    val out = output[Float]("out")
    val definition = kernel("narrowTuple", params(values, out)) { _ =>
      val state = gpuRange("i", literal(0), literal(4)).map(i => convert.f32ToF64(values(i).read))
        .foldLeft("state", (literal(0), literal(0.0), literal(0.0))) { (s, x) =>
          (s._1 + literal(1), s._2 + x, s._3 + x * x)
        }
      out(literal(0)) := convert.f64ToF32(state._2 / convert.i32ToF64(state._1))
      out(literal(1)) := convert.f64ToF32(state._3, RoundingMode.TowardZero)
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    val source = CudaCodegen.generate(definition).toOption.get.cudaSource
    assert(source.contains("out[0] = ::__double2float_rn("))
    assert(source.contains("out[1] = ::__double2float_rz(state_2);"))
    assert(source.contains("double state_1"))

  test("narrowing is typed and never implicit"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val source: Expr[Float] = literal(1.0f)
      convert.f64ToF32(source)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val source: Expr[Double] = literal(1.0)
      val narrow: Expr[Float] = source
    """).nonEmpty)
    val default = convert.f64ToF32(literal(-0.0))
    assertEquals(default, Convert(literal(-0.0), F32,
      RoundingMode.NearestEven, SaturationMode.NoSaturation, SourceSpan.Unknown))
