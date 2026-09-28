package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class WarpSynchronizationSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
  private val full = literal(UInt.fromBits(-1))

  test("warp synchronization is an explicit UInt-masked statement"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("sync") { val result: Unit = warp.sync(literal(UInt.fromBits(-1))) }
    """), Nil)

  test("warp sync requires an explicit unsigned mask and is not a value expression"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("missing") { warp.sync() }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("signed") { warp.sync(literal(-1)) }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("value") { local("x", warp.sync(literal(UInt.fromBits(-1)))); () }
    """).nonEmpty)

  test("sync emits a qualified statement and maps the call-site source span"):
    val span = SourceSpan("Sync.scala", 4, 1, 4, 20)
    val definition = kernel("syncSource") {
      warp.sync(full)(using summon[BlockBuilder], DslSourcePosition(span))
    }
    assertEquals(definition.ir.body.statements, Vector(WarpBarrier(full, span)))
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("::__syncwarp(0xffffffffu);"))
    assert(!generated.cudaSource.contains("__syncthreads"))
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("empty masks forged types and unbound mask references get staged diagnostics"):
    val empty = kernel("empty") { warp.sync(literal(UInt.fromBits(0))) }
    assertEquals(KernelValidator.validate(empty).errors.map(_.code), Vector(ValidationCode.EmptyWarpMask))
    val forged = empty.ir.copy(body = Block(Vector(WarpBarrier(literal(-1).asInstanceOf[Expr[UInt]]))))
    assertEquals(KernelValidator.validate(forged).errors.map(_.code), Vector(ValidationCode.ExpressionTypeMismatch))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(forged))).isLeft)
    val missing = kernel("missing") { warp.sync(value[UInt]("missing")) }
    assertEquals(KernelValidator.validate(missing).errors.map(_.code), Vector(ValidationCode.UnknownScalarParameter))
    var escaped: Expr[UInt] = full
    val escaping = kernel("escaping") {
      scoped { escaped = let("mask", full) }
      warp.sync(escaped)
    }
    assertEquals(KernelValidator.validate(escaping).errors.map(_.code), Vector(ValidationCode.UnboundLocal))

  test("warp ordering effects are distinct from block barriers and register-only collectives"):
    val sync = EffectAnalysis.statement(WarpBarrier(full))
    assertEquals(sync, EffectSummary(hasWarpCollective = true, hasWarpBarrier = true))
    assert(!sync.isPure && !sync.hasBarrier)
    assert(!EffectSummary(hasWarpBarrier = true).isPure)
    val vote = EffectAnalysis.statement(WarpVote(LocalVariable("vote", Bool), WarpVoteOperator.Any, full, literal(true)))
    assert(vote.hasWarpCollective && !vote.hasWarpBarrier && !vote.hasBarrier)
    val both = sync ++ EffectAnalysis.statement(Barrier())
    assert(both.hasBarrier && both.hasWarpCollective && both.hasWarpBarrier)
    assertEquals(EffectSummary.empty ++ sync, sync)
    assertEquals(sync ++ EffectSummary.empty, sync)
    val masks = input[UInt]("masks")
    assertEquals(EffectAnalysis.statement(WarpBarrier(masks(threadIdx.x).read)).readSpaces,
      Set(EffectMemorySpace.Global))

  test("normalization preserves both sides of a memory exchange and repeated barriers"):
    val definition = kernel("ordered") {
      val tile = sharedArray[Int]("tile", 32)
      tile(threadIdx.x) := literal(1)
      warp.sync(full)
      val snapshot = let("snapshot", tile(literal(0)).read)
      warp.sync(full)
      tile(threadIdx.x) := snapshot
      warp.sync(full)
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body, definition.ir.body)
    assertEquals(IrNormalizer.kernel(normalized), normalized)
    assert(KernelValidator.validate(normalized).isValid)

  test("mask expressions normalize and CSE retains their references and names"):
    val masks = input[UInt]("flight4s_cse_0")
    val n = value[Int]("n")
    val definition = kernel("maskCse", params(masks, n)) { _ =>
      warp.sync(masks((n * literal(2)) + (n * literal(2))).read)
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    val temporary = normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]]
    assertEquals(temporary.local.name, "flight4s_cse_1")
    assert(normalized.body.statements.last.isInstanceOf[WarpBarrier])
    assert(KernelValidator.validate(normalized).isValid)
    val folded = kernel("folded") { warp.sync(choose(literal(true))(full)(literal(UInt.fromBits(1)))) }
    assertEquals(IrNormalizer.kernel(folded.ir).body.statements, Vector(WarpBarrier(full)))

  test("masked warp synchronization does not pretend to be a block-wide barrier"):
    val definition = kernel("subgroup") {
      when(threadIdx.x < literal(16)) { warp.sync(literal(UInt.fromBits(0xffff))) }
      when(threadIdx.x < literal(16)) { barrier() }
    }
    val result = KernelValidator.validate(definition)
    assert(result.isValid)
    assertEquals(result.warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))
    assertEquals(result.warnings.head.location, "body.statements[1].then.statements[0]")
    val scope = UniformityScope.empty.withLocal("varying", Uniformity.Varying)
    assertEquals(UniformityAnalysis.scopeAfter(WarpBarrier(full), scope), scope)

  test("warp synchronization is rejected inside expression-only callbacks"):
    val failure = intercept[DslError] {
      kernel("pure") { choose(literal(true)) { warp.sync(full); literal(1) }(literal(0)); () }
    }
    assertEquals(failure.code, DslErrorCode.StatementInsideExpression)

  test("warp synchronization accepts dynamic masks in explicit staged loops"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("loop", params(value[UInt]("mask"))) { p =>
        gpuRange("iteration", literal(0), literal(4)).foreach { _ => warp.sync(p._1) }
      }
    """), Nil)
