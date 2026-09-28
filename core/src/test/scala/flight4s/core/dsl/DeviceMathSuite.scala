package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class DeviceMathSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("device math composes typed Float and Double expressions"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val x = literal(2.0f)
      val y = literal(2.0)
      val a = exp(x) + log(x) + sqrt(x) + rsqrt(x) + tanh(x)
      val b = exp(y) + log(y) + sqrt(y) + rsqrt(y) + tanh(y)
    """), Nil)

  test("every operation emits the standard globally-qualified Float and Double math function"):
    val x = value[Float]("expf")
    val y = value[Double]("exp")
    val floats = output[Float]("floats")
    val doubles = output[Double]("doubles")
    val definition = kernel("deviceMath", params(x, y, floats, doubles)) { _ =>
      UnaryMathOperator.values.zipWithIndex.foreach { (operation, index) =>
        floats(literal(index)) := UnaryMath(operation, x, F32)
        doubles(literal(index)) := UnaryMath(operation, y, F64)
      }
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    UnaryMathOperator.values.zipWithIndex.foreach { (operation, index) =>
      assert(generated.cudaSource.contains(s"floats[$index] = ::${operation.cudaName}f(expf);"))
      assert(generated.cudaSource.contains(s"doubles[$index] = ::${operation.cudaName}(exp);"))
    }
    assertEquals(generated.compilerOptions.additionalNvrtcOptions, Vector.empty)
    assert(!generated.cudaSource.contains("__expf"))

  test("integer Boolean and low-precision operands require explicit supported conversion"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      exp(literal(1))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      sqrt(literal(true))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.Float16
      tanh(literal(Float16.fromBits(0x3c00.toShort)))
    """).nonEmpty)

  test("validation descends into math values and rejects forged operand metadata"):
    val out = output[Float]("out")
    val invalid = kernel("unknownMathInput", params(out)) { _ =>
      out(literal(0)) := exp(input[Float]("missing")(literal(0)).read)
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnknownBuffer))
    val malformed = kernel("badMathType", params(out)) { _ =>
      out(literal(0)) := UnaryMath(UnaryMathOperator.Exp, literal(1).asInstanceOf[Expr[Float]], F32)
    }
    assertEquals(KernelValidator.validate(malformed).errors.map(_.code), Vector(ValidationCode.ExpressionTypeMismatch))
    assert(CudaCodegen.generate(malformed).isLeft)

  test("math retains operand effects uniformity and barrier divergence"):
    val source = input[Float]("source")
    assertEquals(EffectAnalysis.expression(exp(source(literal(0)).read)),
      EffectSummary(readSpaces = Set(EffectMemorySpace.Global)))
    assertEquals(UniformityAnalysis.expression(sqrt(value[Double]("uniform"))), Uniformity.GridUniform)
    assertEquals(UniformityAnalysis.expression(log(source(literal(0)).read)), Uniformity.Varying)
    val definition = kernel("mathBarrier", params(source)) { _ =>
      when(exp(source(threadIdx.x).read) > literal(1.0f)) { barrier() }
    }
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge))

  test("normalization traverses operands but never evaluates CUDA math on the JVM"):
    val span = SourceSpan("Math.scala", 5, 2, 5, 30)
    val input = choose(literal(true))(literal(-0.0f))(literal(4.0f))
    val original = UnaryMath(UnaryMathOperator.Sqrt, input, F32, span)
    val expected = original.copy(value = literal(-0.0f))
    val normalized = IrNormalizer.expression(original)
    assertEquals(normalized, expected)
    assertEquals(IrNormalizer.expression(normalized), normalized)
    for operation <- UnaryMathOperator.values do
      val expression = UnaryMath(operation, literal(0.0), F64, span)
      assertEquals(IrNormalizer.expression(expression), expression)
    val selected = choose(literal(true))(exp(literal(1.0f)))(literal(0.0f))(
      using F32, DslSourcePosition(span)
    )
    assertEquals(IrNormalizer.expression(selected).span, span)
    assertEquals(exp(literal(1.0f))(using F32, DslSourcePosition(span)).span, span)

  test("math traversal retains low-precision headers and reduction temporary names"):
    val out = output[Float]("out")
    val definition = kernel("mathNested", params(out)) { _ =>
      out(literal(0)) := exp(convert.f16ToF32(convert.f32ToF16(literal(1.0f))))
      out(literal(1)) := log(reduceSum("flight4s_accumulator_0", literal(0), literal(2), literal(0.0f))(
        _ => literal(1.0f)
      ))
    }
    val generated = CudaCodegen.generate(definition).toOption.get.cudaSource
    assert(generated.contains("#include <cuda_fp16.h>"))
    assert(generated.contains("float flight4s_accumulator_1 = 0x0.0p0f"), generated)

  test("math CSE traversal reserves nested identifiers without hoisting guarded calls"):
    val flag = value[Boolean]("flag")
    val source = input[Float]("source")
    val out = output[Float]("out")
    val definition = kernel("mathCse", params(flag, source, out)) { _ =>
      val index = threadIdx.x + literal(2)
      val reduction = reduceSum("flight4s_cse_0", literal(0), literal(2), literal(0.0f))(_ => literal(1.0f))
      out(index * index) := exp(reduction)
      out(literal(0)) := choose(flag)(log(source(literal(0)).read))(literal(0.0f))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].local.name, "flight4s_cse_1")
    assertEquals(normalized.body.statements.last, definition.body.statements.last)
    assert(KernelValidator.validate(normalized).isValid)

  test("low precision math accepts explicit accumulator promotion"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.Float16
      val half = literal(Float16.fromBits(0x3c00.toShort))
      val result = exp(half.toAccumulator[Float])
    """), Nil)
