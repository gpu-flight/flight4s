package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class WarpShuffleSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
  private val full = literal(UInt.fromBits(-1))

  test("shared mask validation preserves the existing vote diagnostic order"):
    val invalid = WarpVote(LocalVariable("result", Bool), WarpVoteOperator.Any,
      literal(-1).asInstanceOf[Expr[UInt]], value[Boolean]("missing"))
    val definition = kernel("badVote") {}.ir.copy(body = Block(Vector(invalid)))
    assertEquals(KernelValidator.validate(definition).errors.map(_.code),
      Vector(ValidationCode.UnknownScalarParameter, ValidationCode.ExpressionTypeMismatch))

  test("direct warp shuffle supports typed Int UInt Float and Double snapshots"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      kernel("shuffle") {
        val mask = literal(UInt.fromBits(-1))
        val i: Expr[Int] = warp.shuffle("i", mask, threadIdx.x, literal(0))
        val u: Expr[UInt] = warp.shuffle("u", mask, literal(UInt.fromBits(1)), literal(0))
        val f: Expr[Float] = warp.shuffle("f", mask, literal(1.0f), literal(0))
        val d: Expr[Double] = warp.shuffle("d", mask, literal(1.0), literal(0))
      }
    """), Nil)

  test("shuffle emits one qualified typed declaration and returns a source-located snapshot"):
    val span = SourceSpan("Shuffle.scala", 8, 1, 8, 60)
    val out = output[Float]("out")
    var captured: Expr[Float] = literal(0.0f)
    val definition = kernel("shuffleSource", params(out)) { _ =>
      captured = warp.shuffle("received", full, literal(2.0f), literal(1), 8)(
        using F32, summon[BlockBuilder], DslSourcePosition(span))
      out(literal(0)) := captured
      out(literal(1)) := captured
    }
    assertEquals(captured, Load(LocalVariable("received", F32, span), span))
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("float received = ::__shfl_sync(0xffffffffu, 0x1.0p1f, 1, 8);"))
    assertEquals(generated.cudaSource.sliding("::__shfl_sync".length).count(_ == "::__shfl_sync"), 1)
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("validation accepts exactly six widths and nonnegative source-lane wraparound"):
    for width <- -2 to 40 do
      val definition = kernel("width") { warp.shuffle("result", full, literal(1), literal(33), width); () }
      assertEquals(KernelValidator.validate(definition).isValid, Set(1, 2, 4, 8, 16, 32).contains(width), s"width $width")
    val negative = kernel("negative") { warp.shuffle("result", full, literal(1), literal(-1)); () }
    assertEquals(KernelValidator.validate(negative).errors.map(_.code), Vector(ValidationCode.NegativeWarpSourceLane))
    val empty = kernel("empty") { warp.shuffle("result", literal(UInt.fromBits(0)), literal(1), literal(0)); () }
    assertEquals(KernelValidator.validate(empty).errors.map(_.code), Vector(ValidationCode.EmptyWarpMask))

  test("typing rejects unsupported data masks selectors and writable result use"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.shuffle("x", literal(UInt.fromBits(-1)), literal(true), literal(0)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.{UInt, Float16}
      kernel("bad") { warp.shuffle("x", literal(UInt.fromBits(-1)), literal(Float16.fromBits(0)), literal(0)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { warp.shuffle("x", literal(-1), literal(1), literal(0)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.shuffle("x", literal(UInt.fromBits(-1)), literal(1), literal(0.0f)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { val x = warp.shuffle("x", literal(UInt.fromBits(-1)), literal(1), literal(0)); x := literal(3) }
    """).nonEmpty)

  test("manual shuffle metadata is validated without premature scalar casts"):
    val malformed = WarpShuffle(LocalVariable("result", I32),
      literal(-1).asInstanceOf[Expr[UInt]], literal(1.0f).asInstanceOf[Expr[Int]],
      WarpShuffleOperator.Direct, literal(0.0f).asInstanceOf[Expr[Int]], 32, I32)
    val definition = kernel("malformed") {}.ir.copy(body = Block(Vector(malformed)))
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector(
      ValidationCode.ExpressionTypeMismatch, ValidationCode.ExpressionTypeMismatch,
      ValidationCode.LocalTypeMismatch, ValidationCode.ExpressionTypeMismatch
    ))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(definition))).isLeft)

  test("shuffle results cannot escape lexical scope or bind duplicate names"):
    val out = output[Int]("out")
    var escaped: Expr[Int] = literal(0)
    val definition = kernel("escaping", params(out)) { _ =>
      scoped { escaped = warp.shuffle("received", full, threadIdx.x, literal(0)) }
      out(threadIdx.x) := escaped
    }
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val duplicate = kernel("duplicate") {
      warp.shuffle("result", full, literal(1), literal(0))
      warp.shuffle("result", full, literal(2), literal(0))
      ()
    }
    assertEquals(KernelValidator.validate(duplicate).errors.map(_.code), Vector(ValidationCode.DuplicateLocalName))

  test("shuffle effects account for every operand and remain collective not a memory fence"):
    val masks = input[UInt]("masks")
    val values = input[Float]("values")
    val definition = kernel("effects", params(masks, values)) { _ =>
      val lanes = sharedArray[Int]("lanes", 32)
      warp.shuffle("result", masks(threadIdx.x).read, values(threadIdx.x).read, lanes(threadIdx.x).read)
      ()
    }
    val effects = EffectAnalysis.block(definition.ir.body)
    assertEquals(effects.readSpaces, Set(EffectMemorySpace.Global, EffectMemorySpace.Shared))
    assertEquals(effects.writtenSpaces, Set(EffectMemorySpace.Local))
    assert(effects.hasWarpCollective && !effects.hasBarrier && !effects.isPure)

  test("normalization retains unused width-one shuffles and repeated reads of the result"):
    val definition = kernel("preserve") {
      val result = warp.shuffle("result", full, literal(1) + literal(2), literal(0), width = 1)
      local("twice", result + result)
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.head,
      WarpShuffle(LocalVariable("result", I32), full, literal(3), WarpShuffleOperator.Direct, literal(0), 1, I32))
    assertEquals(normalized.body.statements.last, definition.ir.body.statements.last)
    assertEquals(IrNormalizer.kernel(normalized), normalized)
    assert(KernelValidator.validate(normalized).isValid)

  test("shuffle results remain varying and dependent block barriers remain diagnosed"):
    val definition = kernel("varying") {
      val result = warp.shuffle("result", full, literal(1), threadIdx.x)
      when(result > literal(0)) { barrier() }
    }
    assertEquals(UniformityAnalysis.scopeAfter(definition.ir.body.statements.head, UniformityScope.empty)
      .locals("result"), Uniformity.Varying)
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("CSE shares pure operands but reserves the shuffle result name"):
    val n = value[Int]("n")
    val definition = kernel("cse", params(n)) { _ =>
      warp.shuffle("flight4s_cse_0", full, n * literal(2), n * literal(2))
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    val temporary = normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].local
    assertEquals(temporary.name, "flight4s_cse_1")
    val shuffle = normalized.body.statements.last.asInstanceOf[WarpShuffle[Int, Int]]
    assertEquals(shuffle.value, temporary.read)
    assertEquals(shuffle.selector, temporary.read)
    assert(KernelValidator.validate(normalized).isValid)

  test("shuffle cannot be hidden inside a pure expression callback"):
    val failure = intercept[DslError] {
      kernel("badCallback") { choose(literal(true))(warp.shuffle("result", full, literal(1), literal(0)))(literal(0)); () }
    }
    assertEquals(failure.code, DslErrorCode.StatementInsideExpression)

  test("shuffle accepts a dynamic source lane and explicit subgroup width"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("shuffleWidth") {
        val value = warp.shuffle("value", literal(UInt.fromBits(-1)), threadIdx.x,
          threadIdx.x % literal(8), width = 8)
      }
    """), Nil)
