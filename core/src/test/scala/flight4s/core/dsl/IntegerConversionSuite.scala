package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class IntegerConversionSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("integer inputs have explicit Float and Double conversion helpers"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      val i = literal(3)
      val u = literal(UInt.fromBits(-1))
      val a = convert.i32ToF32(i)
      val b = convert.u32ToF32(u)
      val c = convert.i32ToF64(i)
      val d = convert.u32ToF64(u)
    """), Nil)

  test("all integer conversions emit explicit globally-qualified CUDA intrinsics"):
    val i = value[Int]("signedValue")
    val u = value[UInt]("unsignedValue")
    val floats = output[Float]("floats")
    val doubles = output[Double]("doubles")
    val definition = kernel("integerConversions", params(i, u, floats, doubles)) { _ =>
      RoundingMode.values.zipWithIndex.foreach { (mode, index) =>
        floats(literal(index)) := convert.i32ToF32(i, mode)
        floats(literal(index + 4)) := convert.u32ToF32(u, mode)
      }
      doubles(literal(0)) := convert.i32ToF64(i)
      doubles(literal(1)) := convert.u32ToF64(u)
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    for suffix <- Vector("rn", "rz", "ru", "rd") do
      assert(generated.cudaSource.contains(s"::__int2float_$suffix(signedValue)"))
      assert(generated.cudaSource.contains(s"::__uint2float_$suffix(unsignedValue)"))
    assert(generated.cudaSource.contains("::__int2double_rn(signedValue)"))
    assert(generated.cudaSource.contains("::__uint2double_rn(unsignedValue)"))
    assertEquals(generated.compilerOptions.additionalNvrtcOptions, Vector.empty)

  test("new helpers retain policy types and source spans in existing Convert IR"):
    val span = SourceSpan("Numeric.scala", 5, 2, 5, 55)
    val signed = convert.i32ToF32(literal(16777217), RoundingMode.TowardPositive)(using DslSourcePosition(span))
    assertEquals(signed, Convert(literal(16777217), F32, RoundingMode.TowardPositive, SaturationMode.NoSaturation, span))
    val unsigned = convert.u32ToF64(literal(UInt.fromBits(-1)))(using DslSourcePosition(span))
    assertEquals(unsigned, Convert(literal(UInt.fromBits(-1)), F64, RoundingMode.NearestEven, SaturationMode.NoSaturation, span))

  test("normalization simplifies operands but does not round integer conversions on the JVM"):
    val expression = convert.i32ToF32(literal(16777216) + literal(1), RoundingMode.TowardPositive)
    assertEquals(IrNormalizer.expression(expression), convert.i32ToF32(literal(16777217), RoundingMode.TowardPositive))
    assertEquals(IrNormalizer.expression(convert.i32ToF64(literal(Int.MaxValue))), convert.i32ToF64(literal(Int.MaxValue)))
    val source = input[Int]("source")
    assertEquals(EffectAnalysis.expression(convert.i32ToF32(source(literal(0)).read)).readSpaces,
      Set(EffectMemorySpace.Global))
    assertEquals(UniformityAnalysis.expression(convert.i32ToF32(threadIdx.x)), Uniformity.Varying)

  test("runtime integer lengths compose into guarded floating-point normalization formulas"):
    val n = value[Int]("n")
    val out = output[Float]("out")
    val definition = kernel("runtimeMean", params(n, out)) { _ =>
      val sum = gpuRange("i", literal(0), n).map(i => convert.i32ToF32(i)).sum(literal(0.0f))
      out(literal(0)) := choose(n > literal(0))(sum / convert.i32ToF32(n))(literal(0.0f))
    }
    assert(KernelValidator.validate(definition).isValid)
    assert(CudaCodegen.generate(definition).isRight)

  test("integer conversion helpers reject mismatched source types and implicit conversions"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      convert.i32ToF32(literal(1.0f))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      convert.u32ToF64(literal(1))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val count: Expr[Int] = literal(3)
      literal(1.0f) / count
    """).nonEmpty)

  test("integer to Float conversion accepts explicit rounding intent"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.RoundingMode
      import flight4s.core.types.UInt
      convert.i32ToF32(literal(16777217), RoundingMode.TowardPositive)
      convert.u32ToF32(literal(UInt.fromBits(-1)), RoundingMode.TowardZero)
    """), Nil)
