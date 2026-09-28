package flight4s.core.ir

import munit.FunSuite
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.dsl.DslSourcePosition
import flight4s.core.types.I32

class LoopBoundNormalizationSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("normalization must not freeze an upper-bound local modified by the loop"):
    val definition = kernel("changing_limit") {
      val limit = local("limit", literal(8))
      gpuFor("i", literal(0), limit.read) { _ => limit := literal(0) }
    }
    val original = definition.body.statements(1).asInstanceOf[ForLoop]
    val normalized = IrNormalizer.kernel(definition.ir).body.statements(1).asInstanceOf[ForLoop]
    assertEquals(normalized.until, original.until)

  test("read-only upper bounds still propagate constants"):
    val definition = kernel("fixed_limit") {
      val limit = local("limit", literal(8))
      gpuFor("i", literal(0), limit.read) { _ => () }
    }
    val normalized = IrNormalizer.kernel(definition.ir).body.statements(1).asInstanceOf[ForLoop]
    assertEquals(normalized.until, literal(8))

  test("control-dependence write discovery retains the same visible-local set"):
    val a = LocalVariable("a", I32)
    val b = LocalVariable("b", I32)
    val c = LocalVariable("c", I32)
    val untouched = LocalVariable("untouched", I32)
    val body = Block(Vector(
      Store(a, literal(1)),
      ScopedBlock(Block(Vector(Accumulate(b, literal(1), I32)))),
      ForLoop(LoopIndex("i"), literal(0), literal(2), Block(Vector(Store(c, literal(3))))),
      LocalDeclaration(LocalVariable("temporary", I32), literal(0)),
      Barrier()
    ))
    val scope = UniformityScope(locals = Vector(a, b, c, untouched).map(_.name -> Uniformity.GridUniform).toMap)
    val branch = IfThen(threadIdx.x < literal(1), body, None)
    val after = UniformityAnalysis.scopeAfter(branch, scope)
    assertEquals(after.locals.filter(_._2 == Uniformity.Varying).keySet, Set("a", "b", "c"))
    assertEquals(after.locals("untouched"), Uniformity.GridUniform)
    assertEquals(after.locals.keySet, scope.locals.keySet)
    assertEquals(EffectAnalysis.modifiedLocalNames(body), Set("a", "b", "c"))

  test("writes inside either branch nested scope or nested loop invalidate the upper-bound constant"):
    val flag = value[Boolean]("flag")
    val definition = kernel("nested", params(flag)) { _ =>
      val limit = local("limit", literal(8))
      gpuFor("i", literal(0), limit.read) { _ =>
        gpuIf(flag) {
          scoped { accumulate(limit, literal(-1)) }
        } {
          gpuFor("j", literal(0), literal(1)) { _ => limit := literal(0) }
        }
      }
    }
    val original = definition.body.statements(1).asInstanceOf[ForLoop]
    val normalized = IrNormalizer.kernel(definition.ir).body.statements(1).asInstanceOf[ForLoop]
    assertEquals(normalized.until, original.until)
    assertEquals(EffectAnalysis.modifiedLocalNames(original.body), Set("limit"))
    assert(KernelValidator.validate(IrNormalizer.kernel(definition.ir)).isValid)

  test("compound bounds retain changing reads but still fold invariant locals"):
    val definition = kernel("compound") {
      val limit = local("limit", literal(8))
      val extra = local("extra", literal(2))
      gpuFor("i", literal(0), limit.read + extra.read) { _ => limit := literal(0) }
    }
    val original = definition.body.statements(2).asInstanceOf[ForLoop].until.asInstanceOf[Binary[Int]]
    val normalized = IrNormalizer.kernel(definition.ir).body.statements(2).asInstanceOf[ForLoop]
    assertEquals(normalized.until, original.copy(right = literal(2)))

  test("lower bounds still capture their initial value when the body changes that local"):
    val definition = kernel("lower") {
      val start = local("start", literal(2))
      gpuFor("i", start.read, literal(8)) { _ => start := literal(0) }
    }
    val normalized = IrNormalizer.kernel(definition.ir).body.statements(1).asInstanceOf[ForLoop]
    assertEquals(normalized.from, literal(2))

  test("snapshot locals remain independent of later writes to their source"):
    val definition = kernel("snapshot") {
      val limit = local("limit", literal(8))
      val snapshot = let("snapshot", limit.read)
      gpuFor("i", literal(0), snapshot) { _ => limit := literal(0) }
    }
    val normalized = IrNormalizer.kernel(definition.ir).body.statements(2).asInstanceOf[ForLoop]
    assertEquals(normalized.until, literal(8))

  test("buffer and local-array writes do not invalidate an unrelated invariant bound"):
    val out = output[Int]("out")
    val definition = kernel("memory", params(out)) { _ =>
      val limit = local("limit", literal(8))
      val scratch = localArray[Int]("scratch", 8)
      gpuFor("i", literal(0), limit.read) { i =>
        scratch(i) := i
        out(i) := scratch(i).read
      }
    }
    val original = definition.body.statements(2).asInstanceOf[ForLoop]
    assertEquals(EffectAnalysis.modifiedLocalNames(original.body), Set.empty[String])
    val normalized = IrNormalizer.kernel(definition.ir).body.statements(2).asInstanceOf[ForLoop]
    assertEquals(normalized.until, literal(8))
