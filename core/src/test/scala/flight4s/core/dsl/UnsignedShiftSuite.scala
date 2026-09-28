package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class UnsignedShiftSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("unsigned shifts accept signed device counts and return UInt expressions"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      val word = literal(UInt.fromBits(-1))
      val distance = value[Int]("distance")
      val left: Expr[UInt] = word << distance
      val right: Expr[UInt] = word >> distance
      val logical: Expr[UInt] = word >>> distance
    """), Nil)

  test("shift typing rejects floating values and unsigned floating or Boolean counts"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      literal(1.0f) << literal(2)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      literal(UInt.fromBits(1)) << literal(UInt.fromBits(2))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      literal(UInt.fromBits(1)) >> literal(2.0f)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      literal(UInt.fromBits(1)) >>> literal(true)
    """).nonEmpty)

  test("dynamic CUDA shifts mask signed counts once and right spellings are equivalent"):
    val span = SourceSpan("Shift.scala", 7, 2, 7, 40)
    given DslSourcePosition = DslSourcePosition(span)
    val word = value[UInt]("word")
    val distance = value[Int]("distance")
    val out = output[UInt]("out")
    val left = word << distance
    assertEquals(left.span, span)
    assertEquals(word >> distance, word >>> distance)
    val definition = kernel("shifts", params(word, distance, out)) { _ =>
      out(threadIdx.x) := left | (word >>> distance)
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("((word << (distance & 31)) | (word >> (distance & 31)))"))
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("constant shifts match Scala raw bits for all count boundaries and random pairs"):
    val span = SourceSpan("FoldShift.scala", 3, 1, 3, 40)
    val random = new scala.util.Random(73)
    val words = Vector(0, 1, -1, Int.MinValue, Int.MaxValue, 0x55555555, 0xaaaaaaaa) ++ Vector.fill(25)(random.nextInt())
    val distances = Vector(Int.MinValue, -65, -33, -32, -31, -1, 0, 1, 15, 31, 32, 33, 63, 64, Int.MaxValue) ++ Vector.fill(17)(random.nextInt())
    for bits <- words; distance <- distances do
      for (operator, expected) <- Vector(UnsignedShiftOperator.Left -> (bits << distance), UnsignedShiftOperator.Right -> (bits >>> distance)) do
        val expression = UnsignedShift(operator, literal(UInt.fromBits(bits)), literal(distance), span)
        assertEquals(IrNormalizer.expression(expression), Literal(UInt.fromBits(expected), U32, span))

  test("manual IR validates operand metadata and both reference scopes before lowering"):
    val malformed = UnsignedShift(UnsignedShiftOperator.Left,
      literal(1).asInstanceOf[Expr[UInt]], literal(true).asInstanceOf[Expr[Int]])
    val definition = kernel("malformed") {}.ir.copy(body = Block(Vector(
      LocalDeclaration(LocalVariable("result", U32), malformed)
    )))
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector.fill(2)(ValidationCode.ExpressionTypeMismatch))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(definition))).isLeft)
    val missing = kernel("missing") {
      local("x", value[UInt]("word") << value[Int]("distance")); ()
    }
    assertEquals(KernelValidator.validate(missing).errors.map(_.code), Vector.fill(2)(ValidationCode.UnknownScalarParameter))

  test("all signed count literals are valid and no identity rewrite erases reads"):
    val source = input[UInt]("source")
    val counts = input[Int]("counts")
    val definition = kernel("effects", params(source, counts)) { _ =>
      val tile = sharedArray[Int]("tile", 32)
      val expression = source(threadIdx.x).read << tile(threadIdx.x).read
      assertEquals(EffectAnalysis.expression(expression).readSpaces, Set(EffectMemorySpace.Global, EffectMemorySpace.Shared))
      assertEquals(IrNormalizer.expression(expression), expression)
      val zero = literal(UInt.fromBits(0)) << counts(threadIdx.x).read
      assertEquals(IrNormalizer.expression(zero), zero)
      for distance <- Vector(Int.MinValue, -1, 0, 32, Int.MaxValue) do
        local(s"shift_${distance.toLong + 2147483648L}", source(threadIdx.x).read >> literal(distance))
      ()
    }
    assert(KernelValidator.validate(definition).isValid)

  test("pure integer counts participate in CSE without caching unsigned shifts or reads"):
    val word = value[UInt]("word")
    val n = value[Int]("n")
    val out = output[UInt]("out")
    val definition = kernel("cse", params(word, n, out)) { _ =>
      out(threadIdx.x) := (word << (n + literal(1))) | (word >>> (n + literal(1)))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.size, 2)
    assertEquals(normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].initial, n + literal(1))
    assert(KernelValidator.validate(normalized).isValid)
    assert(CudaCodegen.generate(definition).isRight)

  test("count constant propagation and selected branch spans traverse unsigned shifts"):
    val word = value[UInt]("word")
    val span = SourceSpan("ChooseShift.scala", 9, 1, 9, 50)
    val expression = Conditional(literal(true), word << literal(1), word >> literal(2), U32, span)
    assertEquals(IrNormalizer.expression(expression), UnsignedShift(UnsignedShiftOperator.Left, word, literal(1), span))
    val definition = kernel("propagate") {
      val distance = local("distance", literal(33))
      local("result", literal(UInt.fromBits(-1)) >>> distance.read)
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements(1).asInstanceOf[LocalDeclaration[UInt]].initial,
      literal(UInt.fromBits(Int.MaxValue)))

  test("uniformity includes the count and divergent block barriers remain diagnosed"):
    val one = literal(UInt.fromBits(1))
    assertEquals(UniformityAnalysis.expression(one << threadIdx.x), Uniformity.Varying)
    assertEquals(UniformityAnalysis.expression(value[UInt]("word") >> blockIdx.x), Uniformity.BlockUniform)
    val definition = kernel("divergent") { when((one << threadIdx.x) === one) { barrier() } }
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("unsigned shifts compose through functional map filter and fold"):
    val out = output[UInt]("out")
    val definition = kernel("mask", params(out)) { p =>
      val mask = gpuRange("i", literal(0), literal(32))
        .map(i => literal(UInt.fromBits(1)) << i)
        .filter(bit => (bit & literal(UInt.fromBits(0x55555555))) !== literal(UInt.fromBits(0)))
        .foldLeft("mask", literal(UInt.fromBits(0)))((mask, bit) => mask | bit)
      p._1(threadIdx.x) := mask
    }
    assert(KernelValidator.validate(definition).isValid)
    assert(CudaCodegen.generate(definition).isRight)
