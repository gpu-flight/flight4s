package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class WarpVoteSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
  private val full = literal(UInt.fromBits(-1))

  test("warp votes bind UInt ballots and Boolean all/any results"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      kernel("votes") {
        val mask = literal(UInt.fromBits(-1))
        val ballot: Expr[UInt] = warp.ballot("ballot", mask, threadIdx.x < literal(16))
        val all: Expr[Boolean] = warp.all("all", mask, threadIdx.x < literal(32))
        val any: Expr[Boolean] = warp.any("any", mask, threadIdx.x === literal(0))
      }
    """), Nil)

  test("votes emit one qualified intrinsic each and preserve declaration and snapshot spans"):
    val span = SourceSpan("Votes.scala", 5, 1, 5, 60)
    val out = output[UInt]("out")
    var captured: Expr[UInt] = full
    val definition = kernel("votes", params(out)) { _ =>
      captured = warp.ballot("ballot", full, literal(true))(using summon[BlockBuilder], DslSourcePosition(span))
      warp.all("all", full, literal(true))
      warp.any("any", full, literal(false))
      out(literal(0)) := captured
      out(literal(1)) := captured
    }
    assertEquals(captured, Load(LocalVariable("ballot", U32, span), span))
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("unsigned int ballot = ::__ballot_sync(0xffffffffu, true);"))
    assert(generated.cudaSource.contains("bool all = ::__all_sync(0xffffffffu, true);"))
    assert(generated.cudaSource.contains("bool any = ::__any_sync(0xffffffffu, false);"))
    assertEquals(generated.cudaSource.sliding("::__ballot_sync".length).count(_ == "::__ballot_sync"), 1)
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("typing rejects signed masks non-Boolean predicates and writes to snapshots"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { warp.ballot("x", literal(-1), literal(true)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.any("x", literal(UInt.fromBits(-1)), literal(1)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { val x = warp.all("x", literal(UInt.fromBits(-1)), literal(true)); x := literal(false) }
    """).nonEmpty)

  test("validation rejects empty masks even in branches removed by normalization"):
    val definition = kernel("empty") {
      when(literal(false)) { warp.ballot("result", literal(UInt.fromBits(0)), literal(true)); () }
    }
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector(ValidationCode.EmptyWarpMask))
    assert(CudaCodegen.generate(definition).isLeft)

  test("manual IR checks mask predicate and operator-result type metadata"):
    val vote = WarpVote(LocalVariable("result", I32).asInstanceOf[LocalVariable[UInt]],
      WarpVoteOperator.Ballot, literal(-1).asInstanceOf[Expr[UInt]], literal(1).asInstanceOf[Expr[Boolean]])
    val malformed = kernel("malformed") {}.ir.copy(body = Block(Vector(vote)))
    assertEquals(KernelValidator.validate(malformed).errors.map(_.code), Vector(
      ValidationCode.ExpressionTypeMismatch, ValidationCode.ExpressionTypeMismatch, ValidationCode.LocalTypeMismatch
    ))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(malformed))).isLeft)

  test("vote results obey scope and names and are bound only after operands validate"):
    val duplicate = kernel("duplicate") {
      warp.any("result", full, literal(true))
      warp.all("result", full, literal(false))
      ()
    }
    assertEquals(KernelValidator.validate(duplicate).errors.map(_.code), Vector(ValidationCode.DuplicateLocalName))
    var escaped: Expr[Boolean] = literal(false)
    val badScope = kernel("badScope") {
      scoped { escaped = warp.any("result", full, literal(true)) }
      when(escaped) { barrier() }
    }
    assertEquals(KernelValidator.validate(badScope).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val result = LocalVariable("result", Bool)
    val selfReference = kernel("selfReference") {}.ir.copy(body = Block(Vector(
      WarpVote(result, WarpVoteOperator.Any, full, result.read)
    )))
    assertEquals(KernelValidator.validate(selfReference).errors.map(_.code), Vector(ValidationCode.UnboundLocal))

  test("collective effects compose through blocks without implying a memory barrier"):
    val predicates = input[Boolean]("predicates")
    val definition = kernel("effects", params(predicates)) { _ =>
      scoped { warp.any("result", full, predicates(threadIdx.x).read); () }
    }
    val effects = EffectAnalysis.block(definition.ir.body)
    assertEquals(effects.readSpaces, Set(EffectMemorySpace.Global))
    assertEquals(effects.writtenSpaces, Set(EffectMemorySpace.Local))
    assert(effects.hasWarpCollective)
    assert(!effects.hasBarrier)
    assert(!EffectSummary(hasWarpCollective = true).isPure)
    assert((EffectSummary.empty ++ effects).hasWarpCollective)
    assert((effects ++ EffectSummary.empty).hasWarpCollective)

  test("unused and constant-predicate votes survive normalization as ordered collectives"):
    val definition = kernel("preserve") {
      warp.all("first", full, literal(1) < literal(2))
      warp.any("second", full, literal(false))
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements, Vector(
      WarpVote(LocalVariable("first", Bool), WarpVoteOperator.All, full, literal(true)),
      WarpVote(LocalVariable("second", Bool), WarpVoteOperator.Any, full, literal(false))
    ))
    assertEquals(IrNormalizer.kernel(normalized), normalized)
    assert(KernelValidator.validate(normalized).isValid)

  test("vote results remain conservatively varying for disjoint groups within a warp"):
    val definition = kernel("varying") {
      val result = warp.any("result", full, literal(true))
      when(result) { barrier() }
    }
    val scope = UniformityAnalysis.scopeAfter(definition.ir.body.statements.head, UniformityScope.empty)
    assertEquals(scope.locals("result"), Uniformity.Varying)
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("vote operands participate in pure CSE while result names remain reserved"):
    val n = value[Int]("n")
    val definition = kernel("cse", params(n)) { _ =>
      warp.any("flight4s_cse_0", full, (n * literal(2)) === (n * literal(2)))
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.size, 2)
    assertEquals(normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].local.name, "flight4s_cse_1")
    assert(normalized.body.statements.last.isInstanceOf[WarpVote[?]])
    assert(KernelValidator.validate(normalized).isValid)

  test("collectives cannot be hidden in pure choose arms"):
    val failure = intercept[DslError] {
      kernel("badStaging") { choose(literal(true))(warp.any("result", full, literal(true)))(literal(false)); () }
    }
    assertEquals(failure.code, DslErrorCode.StatementInsideExpression)

  test("warp vote snapshots compose in explicit statement loops"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      val out = output[UInt]("out")
      kernel("votes", params(out)) { _ =>
        gpuRange("i", literal(0), literal(4)).foreach { i =>
          val ballot = warp.ballot("ballot", literal(UInt.fromBits(-1)), threadIdx.x < i)
          out(threadIdx.x * literal(4) + i) := ballot
        }
      }
    """), Nil)
