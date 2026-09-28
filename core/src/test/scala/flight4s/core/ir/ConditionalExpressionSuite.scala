package flight4s.core.ir

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors

import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.{DslSourcePosition, CudaDsl}
import CudaDsl.*
import flight4s.core.types.*

class ConditionalExpressionSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("a literal false conditional selects its false value"):
    val expression = choose(literal(false))(literal(11))(literal(22))
    assertEquals(IrNormalizer.expression(expression), literal(22))

  test("dynamic conditional values generate a guarded CUDA expression"):
    val out = output[Int]("out")
    val definition = kernel("chooseValue", params(out)) { _ =>
      out(literal(0)) := choose(threadIdx.x < literal(4))(literal(11))(literal(22))
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    assert(generated.cudaSource.contains("out[0] = ((threadIdx.x < 4) ? 11 : 22);"))

  test("both result arms must have one CUDA type and the condition must be staged Boolean"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      choose(true)(literal(1))(literal(2))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      choose(literal(1))(literal(1))(literal(2))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      choose(literal(true))(literal(1))(literal(2.0f))
    """).nonEmpty)
    val bool: Expr[Boolean] = choose(literal(true))(literal(false))(literal(true))
    assertEquals(bool.valueType, Bool)

  test("validation checks both arms before pruning and rejects forged runtime types"):
    val out = output[Int]("out")
    val invalid = kernel("invalidArm", params(out)) { _ =>
      out(literal(0)) := choose(literal(true))(literal(1))(input[Int]("missing")(literal(0)).read)
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnknownBuffer))
    assert(CudaCodegen.generate(invalid).isLeft)
    val forged = Conditional(
      literal(1).asInstanceOf[Expr[Boolean]],
      literal(2.0f).asInstanceOf[Expr[Int]], literal(3), I32
    )
    val malformed = kernel("malformedConditional", params(out)) { _ => out(literal(0)) := forged }
    assertEquals(KernelValidator.validate(malformed).errors.map(_.code),
      Vector.fill(2)(ValidationCode.ExpressionTypeMismatch))

  test("effect analysis includes condition reads and both possible result arms"):
    val condition = input[Boolean]("condition")(literal(0)).read
    val left = Load(ConstantElement("constants", literal(0), I32))
    val right = Load(SharedElement("shared", Vector(literal(0)), I32))
    assertEquals(EffectAnalysis.expression(choose(condition)(left)(right)),
      EffectSummary(readSpaces = Set(EffectMemorySpace.Global, EffectMemorySpace.Constant, EffectMemorySpace.Shared)))

  test("conditional values retain control-dependent uniformity and barrier warnings"):
    assertEquals(UniformityAnalysis.expression(choose(threadIdx.x < literal(1))(literal(3))(literal(4))),
      Uniformity.Varying)
    assertEquals(UniformityAnalysis.expression(choose(blockIdx.x < literal(1))(literal(3))(literal(4))),
      Uniformity.BlockUniform)
    assertEquals(UniformityAnalysis.expression(choose(value[Boolean]("flag"))(threadIdx.x)(literal(0))),
      Uniformity.Varying)
    val definition = kernel("conditionalBarrier") {
      val count = local("count", choose(threadIdx.x < literal(1))(literal(1))(literal(2)))
      gpuRange("i", literal(0), count.read).foreach(_ => barrier())
    }
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge))

  test("normalization selects literal conditions and preserves the conditional source span"):
    val span = SourceSpan("Conditional.scala", 4, 3, 4, 70)
    val original = Conditional(literal(1) < literal(2), literal(3) + literal(4), literal(8), I32, span)
    assertEquals(IrNormalizer.expression(original), Literal(7, I32, span))
    val dynamic = Conditional(threadIdx.x < literal(2), literal(3) + literal(4), literal(5) * literal(2), I32, span)
    val normalized = IrNormalizer.expression(dynamic)
    assertEquals(normalized, dynamic.copy(whenTrue = literal(7), whenFalse = literal(10)))
    assertEquals(IrNormalizer.expression(normalized), normalized)
    val positioned = choose(literal(true))(literal(1))(literal(2))(using I32, DslSourcePosition(span))
    assertEquals(positioned.span, span)

  test("equal result arms do not erase an observable condition load"):
    val expression = choose(input[Boolean]("flag")(literal(0)).read)(literal(7))(literal(7))
    assertEquals(IrNormalizer.expression(expression), expression)
    assertEquals(EffectAnalysis.expression(IrNormalizer.expression(expression)), EffectAnalysis.expression(expression))

  test("CSE cannot hoist repeated potentially invalid arithmetic out of a result arm"):
    val divisor = value[Int]("divisor")
    val out = output[Int]("out")
    val definition = kernel("guardedDivide", params(divisor, out)) { _ =>
      val quotient = literal(12) / divisor
      out(literal(0)) := choose(divisor !== literal(0))(quotient + quotient)(literal(9))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body, definition.body)
    assertEquals(IrNormalizer.kernel(normalized), normalized)
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("((divisor != 0) ? ((12 / divisor) + (12 / divisor)) : 9)"))
    assert(!generated.cudaSource.contains("int flight4s_cse_"))

  test("conditional arms retain low-precision headers and nested reduction names"):
    val flag = value[Boolean]("flag")
    val out = output[Float]("out")
    val definition = kernel("conditionalHalf", params(flag, out)) { _ =>
      out(literal(0)) := choose(flag)(
        convert.f16ToF32(convert.f32ToF16(literal(1.0f)))
      )(literal(0.0f))
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("#include <cuda_fp16.h>"))
    val ints = output[Int]("ints")
    val reductionKernel = kernel("conditionalNames", params(flag, ints)) { _ =>
      val nested = reduceSum("flight4s_accumulator_0", literal(0), literal(2), literal(0))(i => i)
      ints(literal(0)) := choose(flag)(nested)(literal(0))
    }
    val source = CudaCodegen.generate(reductionKernel).toOption.get.cudaSource
    assert(source.contains("int flight4s_accumulator_1 = 0"))

  test("CSE reserves identifiers nested inside conditional arms"):
    val flag = value[Boolean]("flag")
    val out = output[Int]("out")
    val definition = kernel("conditionalCseNames", params(flag, out)) { _ =>
      val repeated = threadIdx.x + literal(2)
      val reduction = reduceSum("flight4s_cse_0", literal(0), literal(2), literal(0))(i => i)
      out(literal(0)) := (repeated * repeated) + choose(flag)(reduction)(literal(0))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].local.name, "flight4s_cse_1")
    assert(KernelValidator.validate(normalized).isValid)

  test("normalization preserves selected-arm memory read traces"):
    val flag = input[Boolean]("flag")
    val left = input[Int]("left")
    val right = input[Int]("right")
    val out = output[Int]("out")
    val definition = kernel("conditionalReadTrace", params(flag, left, right, out)) { _ =>
      out(literal(0)) := choose(flag(literal(0)).read)(
        left(literal(0)).read + (literal(1) + literal(2))
      )(right(literal(0)).read + (literal(8) - literal(4)))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    def evaluate(body: Block, selected: Boolean): (Int, Vector[String]) =
      val reads = scala.collection.mutable.ArrayBuffer.empty[String]
      val memory: Map[String, Any] = Map("flag" -> selected, "left" -> 10, "right" -> 20)
      def expression(node: Expr[?]): Any = node match
        case literal: Literal[?] => literal.value
        case load: Load[?, ?, ?] => load.from match
          case buffer: BufferElement[?, ?] =>
            reads += buffer.bufferName
            memory(buffer.bufferName)
          case other => fail(s"unsupported test place: $other")
        case conditional: Conditional[?] =>
          if expression(conditional.condition).asInstanceOf[Boolean] then expression(conditional.whenTrue)
          else expression(conditional.whenFalse)
        case binary: Binary[?] =>
          val a = expression(binary.left).asInstanceOf[Int]
          val b = expression(binary.right).asInstanceOf[Int]
          binary.operator match
            case BinaryOperator.Add => a + b
            case BinaryOperator.Subtract => a - b
            case other => fail(s"unsupported test operator: $other")
        case other => fail(s"unsupported test expression: $other")
      assertEquals(body.statements.size, 1)
      val result = expression(body.statements.head.asInstanceOf[Store[Int, Global]].value).asInstanceOf[Int]
      (result, reads.toVector)
    for selected <- Vector(true, false) do
      val expected = if selected then (13, Vector("flag", "left")) else (24, Vector("flag", "right"))
      assertEquals(evaluate(definition.body, selected), expected)
      assertEquals(evaluate(normalized.body, selected), expected)
