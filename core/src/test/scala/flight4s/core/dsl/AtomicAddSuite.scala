package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class AtomicAddSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("atomicAdd supports typed read-write global memory"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      val ints = inOut[Int]("ints")
      val uints = inOut[UInt]("uints")
      val floats = inOut[Float]("floats")
      kernel("counts", params(ints, uints, floats)) { _ =>
        atomicAdd(ints(literal(0)), literal(1))
        atomicAdd(uints(literal(0)), literal(UInt.fromBits(1)))
        atomicAdd(floats(literal(0)), literal(0.5f))
      }
    """), Nil)

  test("atomicAdd emits a qualified intrinsic with the target address and call-site span"):
    val span = SourceSpan("Histogram.scala", 12, 4, 12, 50)
    val counts = inOut[Int]("counts")
    val definition = kernel("histogram", params(counts)) { _ =>
      val atomicAdd = local("atomicAdd", literal(1))
      CudaDsl.atomicAdd(counts(literal(0)), atomicAdd.read)(using I32, summon[BlockBuilder], DslSourcePosition(span))
    }
    val atomic = definition.ir.body.statements.last.asInstanceOf[AtomicAdd[?, ?]]
    assertEquals(atomic.span, span)
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("::atomicAdd(&counts[0], 1);"))
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("atomicAdd rejects read-only, local, unsupported and mismatched types at compile time"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val data = input[Int]("data")
      kernel("bad", params(data)) { _ => atomicAdd(data(literal(0)), literal(1)) }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { val x = local("x", literal(0)); atomicAdd(x, literal(1)) }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { val x = localArray[Int]("x", 2); atomicAdd(x(literal(0)), literal(1)) }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val data = inOut[Double]("data")
      kernel("bad", params(data)) { _ => atomicAdd(data(literal(0)), literal(1.0)) }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val data = inOut[Boolean]("data")
      kernel("bad", params(data)) { _ => atomicAdd(data(literal(0)), literal(true)) }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val data = inOut[Int]("data")
      kernel("bad", params(data)) { _ => atomicAdd(data(literal(0)), literal(1.0f)) }
    """).nonEmpty)

  test("validation rejects forged local targets and read-only buffer references"):
    val localTarget = LocalVariable("localTarget", I32)
    val badLocal = kernel("badLocal") {}.ir.copy(body = Block(Vector(
      LocalDeclaration(localTarget, literal(0)), AtomicAdd(localTarget, literal(1), I32)
    )))
    assertEquals(KernelValidator.validate(badLocal).errors.map(_.code), Vector(ValidationCode.InvalidAtomicAddressSpace))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(badLocal))).isLeft)
    val readOnly = input[Int]("data")
    val forged = inOut[Int]("data")
    val badAccess = kernel("badAccess", params(readOnly)) { _ => atomicAdd(forged(literal(0)), literal(1)) }
    assertEquals(KernelValidator.validate(badAccess).errors.map(_.code), Vector(ValidationCode.WriteToReadOnlyBuffer))
    assert(CudaCodegen.generate(badAccess).isLeft)

  test("validation checks both atomic value and capability metadata before optimization"):
    val data = inOut[Int]("data")
    val original = kernel("badMetadata", params(data)) { _ => atomicAdd(data(literal(0)), literal(1)) }
    val invalid = AtomicAdd(data(literal(0)), literal(1.0f).asInstanceOf[Expr[Int]],
      F32.asInstanceOf[AtomicAddType[Int]])
    val malformed = original.ir.copy(body = Block(Vector(IfThen(literal(false), Block(Vector(invalid))))))
    assertEquals(KernelValidator.validate(malformed).errors.map(_.code),
      Vector.fill(2)(ValidationCode.ExpressionTypeMismatch))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(malformed))).isLeft)

  test("atomic effects include target reads and writes plus address and value dependencies"):
    val indices = input[Int]("indices")
    val data = inOut[Int]("data")
    val localValue = LocalVariable("value", I32)
    val definition = kernel("effects", params(indices, data)) { _ =>
      val shared = sharedArray[Int]("shared", 4)
      atomicAdd(shared(indices(literal(0)).read), localValue.read)
      atomicAdd(data(literal(0)), shared(literal(0)).read)
    }
    val effects = definition.ir.body.statements.map(EffectAnalysis.statement)
    assertEquals(effects.head.readSpaces, Set(EffectMemorySpace.Global, EffectMemorySpace.Shared, EffectMemorySpace.Local))
    assertEquals(effects.head.writtenSpaces, Set(EffectMemorySpace.Shared))
    assertEquals(effects.last.readSpaces, Set(EffectMemorySpace.Global, EffectMemorySpace.Shared))
    assertEquals(effects.last.writtenSpaces, Set(EffectMemorySpace.Global))
    assert(effects.forall(effect => !effect.isPure && !effect.hasBarrier))

  test("normalization preserves atomic count order zero updates and memory reads"):
    val data = inOut[Int]("data")
    val definition = kernel("orderedAtomics", params(data)) { _ =>
      atomicAdd(data(literal(1) - literal(1)), literal(1) - literal(1))
      atomicAdd(data(literal(0)), data(literal(1)).read + data(literal(1)).read)
      atomicAdd(data(literal(1)), literal(3))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements, Vector(
      AtomicAdd(data(literal(0)), literal(0), I32),
      AtomicAdd(data(literal(0)), data(literal(1)).read + data(literal(1)).read, I32),
      AtomicAdd(data(literal(1)), literal(3), I32)
    ))
    assertEquals(IrNormalizer.kernel(normalized), normalized)
    assert(KernelValidator.validate(normalized).isValid)

  test("atomic operands participate in pure integer CSE without removing the atomic operation"):
    val n = value[Int]("n")
    val data = inOut[Int]("data")
    val definition = kernel("atomicCse", params(n, data)) { _ =>
      atomicAdd(data(n * literal(2)), n * literal(2))
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.size, 2)
    val temp = normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].local
    assertEquals(normalized.body.statements.last, AtomicAdd(data(temp.read), temp.read, I32))
    assert(KernelValidator.validate(normalized).isValid)

  test("atomicAdd cannot escape a pure expression callback through a captured builder"):
    val data = inOut[Int]("data")
    val failure = intercept[DslError] {
      kernel("badExpression", params(data)) { _ =>
        choose(literal(true)) {
          atomicAdd(data(literal(0)), literal(1))
          literal(1)
        }(literal(0))
        ()
      }
    }
    assertEquals(failure.code, DslErrorCode.StatementInsideExpression)

  test("atomicAdd is allowed in divergent flow but subsequent barriers remain checked"):
    val data = inOut[Int]("data")
    val definition = kernel("atomicWarnings", params(data)) { _ =>
      when(threadIdx.x < literal(16)) { atomicAdd(data(literal(0)), literal(1)) }
      when(data(literal(0)).read > literal(0)) { barrier() }
    }
    val result = KernelValidator.validate(definition)
    assert(result.isValid)
    assertEquals(result.warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("atomicAdd composes with foreach and shared array elements"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("sharedCounts", params()) { _ =>
        val counts = sharedArray[Int]("counts", 4)
        gpuRange("i", literal(0), literal(4)).foreach { i =>
          atomicAdd(counts(i), literal(1))
        }
      }
    """), Nil)
