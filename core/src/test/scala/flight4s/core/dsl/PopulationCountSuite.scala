package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class PopulationCountSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("population count accepts signed and unsigned expressions and returns Int"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      val signed: Expr[Int] = bits.popCount(literal(-1))
      val unsigned: Expr[Int] = bits.popCount(literal(UInt.fromBits(-1)))
    """), Nil)

  test("population count rejects Boolean floating and low-precision values"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      bits.popCount(literal(true))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      bits.popCount(literal(1.0f))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.Float16
      bits.popCount(literal(Float16.fromBits(1)))
    """).nonEmpty)

  test("population count folds exact raw bits including zero all ones and negative values"):
    val span = SourceSpan("Count.scala", 5, 1, 5, 50)
    given DslSourcePosition = DslSourcePosition(span)
    val random = new scala.util.Random(76)
    val words = Vector(0, 1, -1, Int.MinValue, Int.MaxValue, 0x55555555, 0xaaaaaaaa) ++ Vector.fill(1017)(random.nextInt())
    for word <- words do
      val expected = Literal(Integer.bitCount(word), I32, span)
      assertEquals(IrNormalizer.expression(bits.popCount(literal(word))), expected)
      assertEquals(IrNormalizer.expression(bits.popCount(literal(UInt.fromBits(word)))), expected)
    assertEquals(IrNormalizer.expression(bits.popCount(literal(-1))).asInstanceOf[Literal[Int]].value, 32)

  test("CUDA emits one qualified intrinsic with an explicit unsigned input conversion"):
    val word = value[Int]("word")
    val unsigned = value[UInt]("unsignedWord")
    val out = output[Int]("out")
    val span = SourceSpan("PopCount.scala", 9, 1, 9, 55)
    given DslSourcePosition = DslSourcePosition(span)
    val expression = bits.popCount(word)
    assertEquals(expression.span, span)
    val definition = kernel("count", params(word, unsigned, out)) { _ =>
      out(threadIdx.x) := expression + bits.popCount(unsigned)
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("::__popc(static_cast<unsigned int>(word))"))
    assert(generated.cudaSource.contains("::__popc(static_cast<unsigned int>(unsignedWord))"))
    assertEquals("::__popc".r.findAllIn(generated.cudaSource).length, 2)
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("manual population count IR validates word metadata and reference scope"):
    val malformed = PopulationCount(literal(1.0f).asInstanceOf[Expr[Int]], I32)
    val definition = kernel("bad") {}.ir.copy(body = Block(Vector(LocalDeclaration(LocalVariable("count", I32), malformed))))
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector(ValidationCode.ExpressionTypeMismatch))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(definition))).isLeft)
    val mixed = PopulationCount(literal(UInt.fromBits(1)).asInstanceOf[Expr[Int]], I32)
    val mixedDefinition = definition.copy(body = Block(Vector(LocalDeclaration(LocalVariable("count", I32), mixed))))
    assertEquals(KernelValidator.validate(mixedDefinition).errors.map(_.code), Vector(ValidationCode.ExpressionTypeMismatch))
    val missing = kernel("missing") { local("x", bits.popCount(value[UInt]("word"))); () }
    assertEquals(KernelValidator.validate(missing).errors.map(_.code), Vector(ValidationCode.UnknownScalarParameter))

  test("population count preserves operand memory effects and is not a collective"):
    val source = input[UInt]("source")
    val expression = bits.popCount(source(threadIdx.x).read)
    val effects = EffectAnalysis.expression(expression)
    assertEquals(effects.readSpaces, Set(EffectMemorySpace.Global))
    assert(!effects.hasWarpCollective)
    assert(!effects.hasBarrier)
    assertEquals(IrNormalizer.expression(expression), expression)
    assert(EffectAnalysis.expression(bits.popCount(value[Int]("word"))).isPure)

  test("existing CSE visits pure population count children without caching memory loads"):
    val word = value[Int]("word")
    val out = output[Int]("out")
    val definition = kernel("cse", params(word, out)) { _ =>
      out(threadIdx.x) := bits.popCount(word ^ literal(1)) + bits.popCount(word ^ literal(1))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.size, 2)
    assertEquals(normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].initial, word ^ literal(1))
    assert(KernelValidator.validate(normalized).isValid)
    assert(CudaCodegen.generate(definition).isRight)

  test("population counts traverse selected branches propagation and nested header dependencies"):
    val word = value[Int]("word")
    val span = SourceSpan("SelectCount.scala", 7, 1, 7, 40)
    val selected = Conditional(literal(true), bits.popCount(word), literal(0), I32, span)
    assertEquals(IrNormalizer.expression(selected), PopulationCount(word, I32, span))
    val definition = kernel("headers", params(word)) { _ =>
      val initial = local("initial", bits.popCount(literal(-1)))
      local("recount", bits.popCount(initial.read))
      val choice = choose(convert.f16ToF32(literal(Float16.fromBits(0x3c00.toShort))) > literal(0.0f))(word)(literal(0))
      local("result", bits.popCount(choice))
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements(1).asInstanceOf[LocalDeclaration[Int]].initial, literal(1))
    assert(CudaCodegen.generate(definition).toOption.get.cudaSource.contains("#include <cuda_fp16.h>"))

  test("uniformity follows the operand and divergent block barriers remain diagnosed"):
    assertEquals(UniformityAnalysis.expression(bits.popCount(threadIdx.x)), Uniformity.Varying)
    assertEquals(UniformityAnalysis.expression(bits.popCount(blockIdx.x)), Uniformity.BlockUniform)
    val definition = kernel("varying") { when(bits.popCount(threadIdx.x) === literal(0)) { barrier() } }
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("population count composes with warp vote snapshots and functional range folds"):
    val definition = kernel("composition", params(output[Int]("out"))) { p =>
      val selected = warp.ballot("selected", literal(UInt.fromBits(-1)), (threadIdx.x & literal(1)) === literal(0))
      val total = gpuRange("bit", literal(0), literal(32))
        .map(bit => bits.popCount(selected & (literal(UInt.fromBits(1)) << bit)))
        .filter(count => count > literal(0)).foldLeft("total", literal(0))(_ + _)
      p._1(threadIdx.x) := total + bits.popCount(selected)
    }
    assert(KernelValidator.validate(definition).isValid)
    assert(CudaCodegen.generate(definition).isRight)
