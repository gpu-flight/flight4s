package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class SignedShiftSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("signed Scala shifts accept signed device counts and return Int expressions"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val word = value[Int]("word")
      val count = value[Int]("count")
      val left: Expr[Int] = word << count
      val right: Expr[Int] = word >> count
      val logical: Expr[Int] = word >>> count
    """), Nil)

  test("signed shifts reject unsigned counts floating values and mixed expression assignments"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      literal(1) << literal(UInt.fromBits(2))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      literal(1.0) >> literal(2)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      val word: Expr[Int] = literal(-1)
      val shifted: Expr[UInt] = word >>> literal(1)
    """).nonEmpty)

  test("constant shifts preserve Scala wrap sign extension and logical right bits"):
    val span = SourceSpan("SignedFold.scala", 5, 1, 5, 50)
    val random = new scala.util.Random(74)
    val words = Vector(0, 1, -1, -2, Int.MinValue, Int.MaxValue, 0x55555555, 0xaaaaaaaa) ++ Vector.fill(24)(random.nextInt())
    val distances = Vector(Int.MinValue, -65, -33, -32, -31, -1, 0, 1, 15, 31, 32, 33, 63, 64, Int.MaxValue) ++ Vector.fill(17)(random.nextInt())
    for bits <- words; distance <- distances do
      for (operator, expected) <- Vector(SignedShiftOperator.Left -> (bits << distance),
          SignedShiftOperator.ArithmeticRight -> (bits >> distance), SignedShiftOperator.LogicalRight -> (bits >>> distance)) do
        assertEquals(IrNormalizer.expression(SignedShift(operator, literal(bits), literal(distance), span)),
          Literal(expected, I32, span))

  test("CUDA lowering reads operands once and uses bounded unsigned-to-signed reconstruction"):
    val source = input[Int]("source")
    val counts = input[Int]("counts")
    val out = output[Int]("out")
    val definition = kernel("lowering", params(source, counts, out)) { _ =>
      out(threadIdx.x) := source(threadIdx.x).read >> counts(threadIdx.x).read
    }
    val cuda = CudaCodegen.generate(definition).toOption.get.cudaSource
    assertEquals("source\\[".r.findAllIn(cuda).length, 1)
    assertEquals("counts\\[".r.findAllIn(cuda).length, 1)
    assert(cuda.contains("& 31u"))
    assert(cuda.contains("& 0x80000000u"))
    assert(cuda.contains("? ~(~"))
    assert(cuda.contains("<= 0x7fffffffu ? static_cast<int>"))
    assert(cuda.contains("(-1 - static_cast<int>(~"))
    assert(!cuda.contains("source[static_cast<int>(threadIdx.x)] >>"))

  test("generated shift temporaries avoid all user names including nested operands"):
    val word = value[Int]("flight4s_value_0")
    val distance = value[Int]("flight4s_value_1")
    val out = output[Int]("flight4s_value_2")
    val definition = kernel("names", params(word, distance, out)) { _ =>
      out(threadIdx.x) := (word << distance) >>> (distance >> literal(1))
    }
    val cuda = CudaCodegen.generate(definition).toOption.get.cudaSource
    for suffix <- 0 to 2 do assert(!cuda.contains(s"const unsigned int flight4s_value_$suffix ="))
    val names = "const unsigned int (flight4s_value_[0-9]+) =".r.findAllMatchIn(cuda).map(_.group(1)).toVector
    assertEquals(names.size, 9)
    assertEquals(names.distinct.size, 9)

  test("manual IR rejects wrong operand metadata and unknown references"):
    val expression = SignedShift(SignedShiftOperator.Left, literal(true).asInstanceOf[Expr[Int]],
      literal(UInt.fromBits(1)).asInstanceOf[Expr[Int]])
    val definition = kernel("bad") {}.ir.copy(body = Block(Vector(LocalDeclaration(LocalVariable("x", I32), expression))))
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector.fill(2)(ValidationCode.ExpressionTypeMismatch))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(definition))).isLeft)
    val missing = kernel("missing") { local("x", value[Int]("word") >>> value[Int]("count")); () }
    assertEquals(KernelValidator.validate(missing).errors.map(_.code), Vector.fill(2)(ValidationCode.UnknownScalarParameter))

  test("normalization retains both memory effects even when the value is zero or count is zero"):
    val source = input[Int]("source")
    val definition = kernel("effects", params(source)) { _ =>
      val counts = sharedArray[Int]("counts", 32)
      val expression = source(threadIdx.x).read << counts(threadIdx.x).read
      assertEquals(EffectAnalysis.expression(expression).readSpaces, Set(EffectMemorySpace.Global, EffectMemorySpace.Shared))
      assertEquals(IrNormalizer.expression(expression), expression)
      assertEquals(IrNormalizer.expression(literal(0) >> counts(threadIdx.x).read), literal(0) >> counts(threadIdx.x).read)
      assertEquals(IrNormalizer.expression(source(threadIdx.x).read << literal(0)), source(threadIdx.x).read << literal(0))
      local("x", expression)
      ()
    }
    assert(KernelValidator.validate(definition).isValid)

  test("count CSE traverses signed shifts without adding shift-result CSE"):
    val word = value[Int]("word")
    val count = value[Int]("count")
    val out = output[Int]("out")
    val definition = kernel("cse", params(word, count, out)) { _ =>
      out(threadIdx.x) := (word << (count & literal(31))) | (word >>> (count & literal(31)))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.size, 2)
    assertEquals(normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].initial, count & literal(31))
    assert(KernelValidator.validate(normalized).isValid)
    assert(CudaCodegen.generate(definition).isRight)

  test("constant propagation source spans and selected conditional arms retain signed shift behavior"):
    val span = SourceSpan("Signed.scala", 9, 1, 9, 50)
    given DslSourcePosition = DslSourcePosition(span)
    val word = value[Int]("word")
    val expression = word >> literal(1)
    assertEquals(expression.span, span)
    val selected = Conditional(literal(true), expression, word << literal(2), I32, span)
    assertEquals(IrNormalizer.expression(selected), expression)
    val definition = kernel("propagate") {
      val distance = local("distance", literal(31))
      val shifted = local("shifted", literal(-1) << distance.read)
      local("sign", shifted.read >> distance.read)
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements(1).asInstanceOf[LocalDeclaration[Int]].initial, Literal(Int.MinValue, I32, span))
    assertEquals(normalized.body.statements(2).asInstanceOf[LocalDeclaration[Int]].initial, Literal(-1, I32, span))
    val generated = CudaCodegen.generate(kernel("mapped", params(word)) { _ => local("x", expression); () }).toOption.get
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("functional bit extraction retains varying uniformity and barrier diagnostics"):
    val definition = kernel("bits", params(output[Int]("out"))) { p =>
      val count = gpuRange("bit", literal(0), literal(32))
        .map(bit => (threadIdx.x >>> bit) & literal(1))
        .filter(bit => bit === literal(1)).foldLeft("count", literal(0))(_ + _)
      p._1(threadIdx.x) := count
    }
    assert(CudaCodegen.generate(definition).isRight)
    assertEquals(UniformityAnalysis.expression(literal(1) << threadIdx.x), Uniformity.Varying)
    val divergent = kernel("divergent") { when((threadIdx.x >> literal(1)) === literal(0)) { barrier() } }
    assertEquals(KernelValidator.validate(divergent).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))
