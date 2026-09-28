package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class BitwiseComplementSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("Scala complement supports signed and unsigned device integers"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      val signed: Expr[Int] = ~literal(3)
      val unsigned: Expr[UInt] = ~literal(UInt.fromBits(3))
    """), Nil)

  test("complement rejects Boolean floating and low-precision operands"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      ~literal(true)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      ~literal(1.0f)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.Float16
      ~literal(Float16.fromBits(1))
    """).nonEmpty)

  test("complement builds exactly the established XOR IR with the correct all-ones type"):
    val span = SourceSpan("Complement.scala", 5, 2, 5, 30)
    given DslSourcePosition = DslSourcePosition(span)
    val signed = value[Int]("signed")
    val unsigned = value[UInt]("unsigned")
    assertEquals(~signed, Binary(BinaryOperator.BitXor, signed, Literal(-1, I32, span), I32, span))
    assertEquals(~unsigned, Binary(BinaryOperator.BitXor, unsigned, Literal(UInt.fromBits(-1), U32, span), U32, span))
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.dsl.DslSourcePosition
      import flight4s.core.ir.Expr
      import flight4s.core.types.BitwiseType
      def complement[T](value: Expr[T])(using BitwiseType[T], DslSourcePosition): Expr[T] = ~value
    """), Nil)

  test("generated source is unchanged from explicit XOR for both signednesses"):
    val signed = value[Int]("word")
    val signedOut = output[Int]("out")
    val unary = kernel("complement", params(signed, signedOut)) { _ => signedOut(threadIdx.x) := ~signed }
    val binary = kernel("complement", params(signed, signedOut)) { _ => signedOut(threadIdx.x) := signed ^ literal(-1) }
    assertEquals(CudaCodegen.generate(unary).toOption.get.cudaSource, CudaCodegen.generate(binary).toOption.get.cudaSource)
    val unsigned = value[UInt]("word")
    val unsignedOut = output[UInt]("out")
    val unaryU = kernel("complement", params(unsigned, unsignedOut)) { _ => unsignedOut(threadIdx.x) := ~unsigned }
    val binaryU = kernel("complement", params(unsigned, unsignedOut)) { _ => unsignedOut(threadIdx.x) := unsigned ^ literal(UInt.fromBits(-1)) }
    assertEquals(CudaCodegen.generate(unaryU).toOption.get.cudaSource, CudaCodegen.generate(binaryU).toOption.get.cudaSource)

  test("signed complement folding matches raw bits while unsigned folding stays deferred"):
    val random = new scala.util.Random(75)
    val words = Vector(0, 1, -1, Int.MinValue, Int.MaxValue, 0x55555555, 0xaaaaaaaa) ++ Vector.fill(1017)(random.nextInt())
    for word <- words do
      assertEquals(IrNormalizer.expression(~literal(word)), literal(~word))
      assertEquals(IrNormalizer.expression(~(~literal(word))), literal(word))
    val unsigned = ~literal(UInt.fromBits(-1))
    assertEquals(IrNormalizer.expression(unsigned), unsigned)

  test("complement preserves reads even under double complement and retains validation"):
    val source = input[Int]("source")
    val expression = ~(~source(threadIdx.x).read)
    assertEquals(EffectAnalysis.expression(expression).readSpaces, Set(EffectMemorySpace.Global))
    assertEquals(IrNormalizer.expression(expression), expression)
    val malformed = kernel("malformed") { local("x", ~literal(1.0f).asInstanceOf[Expr[Int]]); () }
    assertEquals(KernelValidator.validate(malformed).errors.map(_.code), Vector(ValidationCode.ExpressionTypeMismatch))
    val missing = kernel("missing") { local("x", ~value[Int]("missing")); () }
    assertEquals(KernelValidator.validate(missing).errors.map(_.code), Vector(ValidationCode.UnknownScalarParameter))

  test("complement retains existing signed CSE source spans and barrier uniformity"):
    val word = value[Int]("word")
    val out = output[Int]("out")
    val definition = kernel("cse", params(word, out)) { _ => out(threadIdx.x) := (~word) & (~word) }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.size, 2)
    assertEquals(normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].initial, ~word)
    assert(KernelValidator.validate(normalized).isValid)
    val varying = kernel("varying") { when((~threadIdx.x) === literal(0)) { barrier() } }
    assertEquals(KernelValidator.validate(varying).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("complement composes with shifts and functional mask filtering"):
    val definition = kernel("inverse_mask", params(output[UInt]("out"))) { p =>
      val allowed = ~literal(UInt.fromBits(0xaaaaaaaa))
      val mask = gpuRange("bit", literal(0), literal(32))
        .map(bit => literal(UInt.fromBits(1)) << bit)
        .filter(bit => (bit & allowed) !== literal(UInt.fromBits(0)))
        .foldLeft("mask", literal(UInt.fromBits(0)))((mask, bit) => mask | bit)
      p._1(threadIdx.x) := mask
    }
    assert(CudaCodegen.generate(definition).isRight)
