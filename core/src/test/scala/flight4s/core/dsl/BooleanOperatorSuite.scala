package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class BooleanOperatorSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("staged Booleans support ordinary Scala logical operator syntax"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val a = value[Boolean]("a")
      val b = value[Boolean]("b")
      val both = a && b
      val either = a || b
      val neither = !(a || b)
    """), Nil)

  test("logical operators lower to the existing conditional IR and identical artifacts"):
    val a = value[Boolean]("a")
    val b = value[Boolean]("b")
    val out = output[Boolean]("out")
    val signature = params(a, b, out)
    val actual = kernel("logicalOperators", signature) { _ =>
      out(literal(0)) := a && b
      out(literal(1)) := a || b
      out(literal(2)) := !a
    }
    val expected = kernel("logicalOperators", signature) { _ =>
      out(literal(0)) := choose(a)(b)(literal(false))
      out(literal(1)) := choose(a)(literal(true))(b)
      out(literal(2)) := choose(a)(literal(false))(literal(true))
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("truth tables and selected memory read traces survive normalization"):
    val a = input[Boolean]("a")(literal(0)).read
    val b = input[Boolean]("b")(literal(0)).read
    def evaluate(node: Expr[Boolean], left: Boolean, right: Boolean): (Boolean, Vector[String]) =
      val reads = scala.collection.mutable.ArrayBuffer.empty[String]
      def loop(expression: Expr[?]): Boolean = expression match
        case literal: Literal[?] => literal.value.asInstanceOf[Boolean]
        case load: Load[?, ?, ?] => load.from match
          case buffer: BufferElement[?, ?] =>
            reads += buffer.bufferName
            if buffer.bufferName == "a" then left else right
          case other => fail(s"unsupported place: $other")
        case conditional: Conditional[?] =>
          if loop(conditional.condition) then loop(conditional.whenTrue) else loop(conditional.whenFalse)
        case other => fail(s"unsupported expression: $other")
      (loop(node), reads.toVector)
    for left <- Vector(false, true); right <- Vector(false, true) do
      val cases = Vector(
        (a && b, left && right, if left then Vector("a", "b") else Vector("a")),
        (a || b, left || right, if left then Vector("a") else Vector("a", "b")),
        (!a, !left, Vector("a")),
        (!(a && b), !(left && right), if left then Vector("a", "b") else Vector("a"))
      )
      for (expression, expected, reads) <- cases do
        assertEquals(evaluate(expression, left, right), (expected, reads))
        assertEquals(evaluate(IrNormalizer.expression(expression), left, right), (expected, reads))

  test("right builders stage once even when the left literal will short-circuit on device"):
    var conjunctions = 0
    var disjunctions = 0
    val and = literal(false) && { conjunctions += 1; literal(true) }
    val or = literal(true) || { disjunctions += 1; literal(false) }
    assertEquals((conjunctions, disjunctions), (1, 1))
    assertEquals(IrNormalizer.expression(and), literal(false))
    assertEquals(IrNormalizer.expression(or), literal(true))

  test("right builders cannot leak stores or barriers into the surrounding block"):
    for conjunction <- Vector(false, true) do
      val error = intercept[DslError] {
        kernel("badLogicalEffect") {
          if conjunction then literal(false) && { barrier(); literal(true) }
          else literal(true) || { barrier(); literal(false) }
          ()
        }
      }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("unselected right expressions are validated before normalization"):
    val out = output[Boolean]("out")
    val invalid = kernel("unknownLogicalInput", params(out)) { _ =>
      out(literal(0)) := literal(false) && input[Boolean]("missing")(literal(0)).read
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnknownBuffer))
    assert(CudaCodegen.generate(invalid).isLeft)

  test("guarded arithmetic stays inside conditional evaluation through CSE"):
    val divisor = value[Int]("divisor")
    val out = output[Boolean]("out")
    val definition = kernel("logicalDivision", params(divisor, out)) { _ =>
      val quotient = literal(12) / divisor
      out(literal(0)) := (divisor !== literal(0)) && quotient + quotient > literal(2)
    }
    assertEquals(IrNormalizer.kernel(definition.ir).body, definition.body)
    val source = CudaCodegen.generate(definition).toOption.get.cudaSource
    assert(source.contains("((divisor != 0) ? (((12 / divisor) + (12 / divisor)) > 2) : false)"))
    assert(!source.contains("int flight4s_cse_"))

  test("logical expressions retain operator source positions effects and varying control"):
    val span = SourceSpan("Logical.scala", 3, 2, 3, 40)
    val left = input[Boolean]("left")(literal(0)).read
    val right = threadIdx.x < literal(1)
    val and = left.&&(right)(using DslSourcePosition(span))
    assertEquals(and.span, span)
    assertEquals(left.||(right)(using DslSourcePosition(span)).span, span)
    assertEquals(left.unary_!(using DslSourcePosition(span)).span, span)
    assertEquals(EffectAnalysis.expression(and).readSpaces, Set(EffectMemorySpace.Global))
    assertEquals(UniformityAnalysis.expression(and), Uniformity.Varying)
    val definition = kernel("logicalBarrier") { when(right && literal(true)) { barrier() } }
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge))

  test("operators require staged Boolean operands and cannot drive host if statements"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      literal(1) && literal(true)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      literal(true) || false
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      !literal(1)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      if literal(true) && literal(false) then ()
    """).nonEmpty)

  test("logical expressions compose inside staged for guards"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("booleanGuard") {
        for i <- gpuRange("i", literal(0), literal(8))
            if i > literal(0) && literal(12) / i > literal(2)
        do barrier()
      }
    """), Nil)
