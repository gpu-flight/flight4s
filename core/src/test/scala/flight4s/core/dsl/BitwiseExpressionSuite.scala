package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class BitwiseExpressionSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("bitwise expressions support Scala operators on signed and unsigned device integers"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      val a: Expr[Int] = (literal(3) & literal(1)) | (literal(2) ^ literal(7))
      val u = literal(UInt.fromBits(-1))
      val b: Expr[UInt] = (u & u) | (u ^ u)
    """), Nil)

  test("typing rejects floating Boolean low-precision and mixed signedness operands"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      literal(1.0f) & literal(2.0f)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      literal(true) | literal(false)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.Float16
      literal(Float16.fromBits(1)) ^ literal(Float16.fromBits(2))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      literal(1) & literal(UInt.fromBits(2))
    """).nonEmpty)

  test("manual bitwise IR accepts exactly I32 and U32 among all scalar types"):
    def check[T](valueType: CudaType[T], value: T): Unit =
      for operator <- Vector(BinaryOperator.BitAnd, BinaryOperator.BitOr, BinaryOperator.BitXor) do
        val expression = Binary(operator, Literal(value, valueType), Literal(value, valueType), valueType)
        val definition = kernel("manual") {}.ir.copy(body = Block(Vector(
          LocalDeclaration(LocalVariable("result", valueType), expression)
        )))
        val expected = if valueType == I32 || valueType == U32 then Vector.empty
          else Vector(ValidationCode.UnsupportedBitwiseType)
        assertEquals(KernelValidator.validate(definition).errors.map(_.code), expected)
        assertEquals(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(definition))).isRight, expected.isEmpty)
    check(I32, 0)
    check(U32, UInt.fromBits(0))
    check(Bool, false)
    check(F32, 0.0f)
    check(F64, 0.0)
    check(F16, Float16.fromBits(0))
    check(BF16, BFloat16.fromBits(0))
    check(FP8E4M3, Float8E4M3.fromBits(0))
    check(FP8E5M2, Float8E5M2.fromBits(0))

  test("malformed operand metadata and unknown references retain validation"):
    val expression = Binary(BinaryOperator.BitAnd, literal(1), literal(1.0f).asInstanceOf[Expr[Int]], I32)
    val definition = kernel("malformed") {}.ir.copy(body = Block(Vector(LocalDeclaration(LocalVariable("x", I32), expression))))
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector(ValidationCode.ExpressionTypeMismatch))
    val missing = kernel("missing") { local("x", value[Int]("missing") & literal(1)); () }
    assertEquals(KernelValidator.validate(missing).errors.map(_.code), Vector(ValidationCode.UnknownScalarParameter))

  test("CUDA emission preserves parentheses and captures bitwise expression source spans"):
    val span = SourceSpan("Bits.scala", 7, 2, 7, 50)
    given DslSourcePosition = DslSourcePosition(span)
    val a = value[Int]("a")
    val b = value[Int]("b")
    val out = output[Int]("out")
    val expression = (a & b) | (a ^ literal(7))
    assertEquals(expression.span, span)
    val definition = kernel("bits", params(a, b, out)) { _ => out(threadIdx.x) := expression }
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("((a & b) | (a ^ 7))"))
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("signed constant folding agrees with Scala for boundary and deterministic random bit patterns"):
    val span = SourceSpan("Fold.scala", 3, 1, 3, 40)
    val random = new scala.util.Random(17)
    val values = Vector(0, 1, -1, Int.MinValue, Int.MaxValue, 0x55555555, 0xaaaaaaaa) ++ Vector.fill(25)(random.nextInt())
    for left <- values; right <- values do
      val cases = Vector(
        BinaryOperator.BitAnd -> (left & right), BinaryOperator.BitOr -> (left | right), BinaryOperator.BitXor -> (left ^ right)
      )
      for (operator, expected) <- cases do
        val expression = Binary(operator, literal(left), literal(right), I32, span)
        assertEquals(IrNormalizer.expression(expression), Literal(expected, I32, span))
    val unsigned = literal(UInt.fromBits(-1)) & literal(UInt.fromBits(1))
    assertEquals(IrNormalizer.expression(unsigned), unsigned)

  test("bitwise identities do not erase memory reads and eager operators retain both dependencies"):
    val source = input[Int]("source")
    val definition = kernel("effects", params(source)) { _ =>
      val tile = sharedArray[Int]("tile", 32)
      val expression = (source(threadIdx.x).read & literal(0)) | tile(threadIdx.x).read
      assertEquals(EffectAnalysis.expression(expression).readSpaces, Set(EffectMemorySpace.Global, EffectMemorySpace.Shared))
      assertEquals(IrNormalizer.expression(expression), expression)
      local("x", expression)
      ()
    }
    assert(KernelValidator.validate(definition).isValid)

  test("pure bitwise integer subexpressions share existing CSE without caching loads"):
    val n = value[Int]("n")
    val out = output[Int]("out")
    val definition = kernel("cse", params(n, out)) { _ =>
      out(threadIdx.x) := (n & literal(7)) + (n & literal(7))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    val temporary = normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]]
    assertEquals(temporary.initial, n & literal(7))
    assertEquals(normalized.body.statements.size, 2)
    assert(KernelValidator.validate(normalized).isValid)

  test("bitwise conditions retain varying uniformity and block barrier diagnostics"):
    assertEquals(UniformityAnalysis.expression(threadIdx.x & literal(31)), Uniformity.Varying)
    val definition = kernel("varying") { when((threadIdx.x & literal(1)) === literal(0)) { barrier() } }
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("bitwise expressions compose in pure functional ranges"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("masked", params(output[Int]("out"))) { p =>
        val total = gpuRange("i", literal(0), literal(32))
          .map(i => i & literal(7)).filter(i => (i ^ literal(1)) > literal(0))
          .foldLeft("total", literal(0))((sum, value) => sum + value)
        p._1(threadIdx.x) := total
      }
    """), Nil)
