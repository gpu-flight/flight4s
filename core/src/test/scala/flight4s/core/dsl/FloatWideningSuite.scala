package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class FloatWideningSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("Float expressions have an explicit Double widening helper"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val wide: Expr[Double] = convert.f32ToF64(literal(1.0f))
    """), Nil)

  test("Float widening is accepted and emits a CUDA double cast"):
    val source = value[Float]("source")
    val out = output[Double]("out")
    val definition = kernel("widen", params(source, out)) { _ =>
      out(literal(0)) := Convert(source, F64, RoundingMode.NearestEven, SaturationMode.NoSaturation)
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("out[0] = static_cast<double>(source);"))
    assertEquals(generated.compilerOptions.additionalNvrtcOptions, Vector.empty)

  test("raw widening IR rejects noncanonical rounding and saturation with precise diagnostics"):
    val span = SourceSpan("Widen.scala", 8, 2, 8, 65)
    val definition = kernel("badWiden") {
      local("wide", Convert(literal(1.0f), F64,
        RoundingMode.TowardZero, SaturationMode.SaturateFinite, span))
      ()
    }
    val errors = KernelValidator.validate(definition).errors
    assertEquals(errors.map(_.code), Vector(
      ValidationCode.UnsupportedConversionRounding, ValidationCode.UnsupportedConversionSaturation))
    assert(errors.forall(_.span == span))
    assert(errors.forall(_.location.endsWith(".initial")))
    assert(CudaCodegen.generate(definition).isLeft)

  test("widening retains source spans and is not evaluated by the JVM normalizer"):
    val span = SourceSpan("Widen.scala", 14, 2, 14, 60)
    val widened = convert.f32ToF64(literal(-0.0f))(using DslSourcePosition(span))
    assertEquals(widened, Convert(literal(-0.0f), F64,
      RoundingMode.NearestEven, SaturationMode.NoSaturation, span))
    assertEquals(IrNormalizer.expression(widened), widened)
    val source = input[Float]("source")
    val widenedLoad = convert.f32ToF64(source(literal(0)).read)
    assertEquals(EffectAnalysis.expression(widenedLoad).readSpaces, Set(EffectMemorySpace.Global))
    assertEquals(UniformityAnalysis.expression(widenedLoad), Uniformity.Varying)
    assertEquals(UniformityAnalysis.expression(convert.f32ToF64(literal(1.0f))), Uniformity.GridUniform)

  test("widened Float values compose with independently typed tuple state"):
    val values = input[Float]("values")
    val out = output[Double]("out")
    val definition = kernel("wideTuple", params(values, out)) { _ =>
      val state = gpuRange("i", literal(0), literal(4)).map(i => convert.f32ToF64(values(i).read))
        .foldLeft("state", (literal(0), literal(0.0), literal(0.0))) { (s, x) =>
          (s._1 + literal(1), s._2 + x, s._3 + x * x)
        }
      out(literal(0)) := state._2 / convert.i32ToF64(state._1)
      out(literal(1)) := state._3
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    val source = CudaCodegen.generate(definition).toOption.get.cudaSource
    assert(source.contains("static_cast<double>(values[i])"))
    assert(source.contains("double state_1"))

  test("widening does not enable implicit promotion or unrelated narrowing"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val source: Expr[Double] = literal(1.0)
      convert.f32ToF64(source)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val source: Expr[Float] = literal(1.0f)
      val wide: Expr[Double] = source
    """).nonEmpty)
    val invalid = kernel("narrow") {
      local("narrowed", Convert(literal(1.0), F32, RoundingMode.NearestEven, SaturationMode.NoSaturation))
      ()
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnsupportedConversion))
