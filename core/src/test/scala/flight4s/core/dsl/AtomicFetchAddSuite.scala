package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class AtomicFetchAddSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("atomicFetchAdd returns a typed snapshot usable more than once"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val counters = inOut[Int]("counters")
      val output = out[Int]("output")
      kernel("tickets", params(counters, output)) { _ =>
        val ticket: Expr[Int] = atomicFetchAdd("ticket", counters(literal(0)), literal(1))
        output(ticket) := ticket
      }
    """), Nil)

  test("capture stages exactly one atomic and subsequent uses read its local snapshot"):
    val span = SourceSpan("Tickets.scala", 9, 2, 9, 50)
    val counters = inOut[Int]("counters")
    val out = output[Int]("out")
    var captured: Expr[Int] = literal(-1)
    val definition = kernel("capture", params(counters, out)) { _ =>
      captured = atomicFetchAdd("ticket", counters(literal(0)), literal(1))(
        using I32, summon[BlockBuilder], DslSourcePosition(span))
      out(literal(0)) := captured
      out(literal(1)) := captured + captured
    }
    val expectedLocal = LocalVariable("ticket", I32, span)
    assertEquals(captured, Load(expectedLocal, span))
    assertEquals(definition.ir.body.statements.head,
      AtomicFetchAdd(expectedLocal, counters(literal(0)), literal(1), I32, span))
    val generated = CudaCodegen.generate(definition).toOption.get
    assertEquals(generated.cudaSource.sliding("::atomicAdd".length).count(_ == "::atomicAdd"), 1)
    assert(generated.cudaSource.contains("int ticket = ::atomicAdd(&counters[0], 1);"))
    assert(generated.cudaSource.contains("out[1] = (ticket + ticket);"))
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("result expressions are read-only and capture rejects invalid targets and types"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val data = inOut[Int]("data")
      kernel("bad", params(data)) { _ =>
        val result = atomicFetchAdd("result", data(literal(0)), literal(1))
        result := literal(2)
      }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val data = input[Int]("data")
      kernel("bad", params(data)) { _ => atomicFetchAdd("x", data(literal(0)), literal(1)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { val data = local("data", literal(0)); atomicFetchAdd("x", data, literal(1)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val data = inOut[Double]("data")
      kernel("bad", params(data)) { _ => atomicFetchAdd("x", data(literal(0)), literal(1.0)); () }
    """).nonEmpty)

  test("capture uses ordinary lexical binding checks and results cannot escape scope"):
    val data = inOut[Int]("data")
    val duplicate = kernel("duplicate", params(data)) { _ =>
      atomicFetchAdd("ticket", data(literal(0)), literal(1))
      atomicFetchAdd("ticket", data(literal(0)), literal(1))
      ()
    }
    assertEquals(KernelValidator.validate(duplicate).errors.map(_.code), Vector(ValidationCode.DuplicateLocalName))
    val collision = kernel("collision", params(data)) { _ => atomicFetchAdd("data", data(literal(0)), literal(1)); () }
    assertEquals(KernelValidator.validate(collision).errors.map(_.code), Vector(ValidationCode.LocalNameConflictsWithBinding))
    var escaped: Expr[Int] = literal(0)
    val escaping = kernel("escaping", params(data)) { _ =>
      scoped { escaped = atomicFetchAdd("ticket", data(literal(0)), literal(1)) }
      data(literal(1)) := escaped
    }
    assertEquals(KernelValidator.validate(escaping).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val siblings = kernel("siblings", params(data)) { _ =>
      for _ <- 0 until 2 do scoped { atomicFetchAdd("ticket", data(literal(0)), literal(1)); () }
    }
    assert(KernelValidator.validate(siblings).isValid)

  test("atomic validation is shared with update-only statements and checks result metadata"):
    val data = input[Int]("data")
    val forged = inOut[Int]("data")
    val span = SourceSpan("Bad.scala", 3, 1, 3, 40)
    val update = AtomicAdd(forged(literal(0)), literal(1.0f).asInstanceOf[Expr[Int]], I32, span)
    val result = LocalVariable("result", I32)
    val capture = AtomicFetchAdd(result, update.target, update.value, update.addition, span)
    val base = kernel("bad", params(data)) { _ => () }.ir
    assertEquals(
      KernelValidator.validate(base.copy(body = Block(Vector(capture)))).errors,
      KernelValidator.validate(base.copy(body = Block(Vector(update)))).errors
    )
    val wrongResult = capture.copy(local = LocalVariable("result", F32).asInstanceOf[LocalVariable[Int]])
    assert(KernelValidator.validate(base.copy(body = Block(Vector(wrongResult)))).errors
      .exists(_.code == ValidationCode.LocalTypeMismatch))
    val selfReference = capture.copy(value = result.read)
    assert(KernelValidator.validate(base.copy(body = Block(Vector(selfReference)))).errors
      .exists(_.code == ValidationCode.UnboundLocal))

  test("invalid atomic operands do not cause spurious unknown-result diagnostics"):
    val declared = inOut[Int]("declared")
    val missing = inOut[Int]("missing")
    val definition = kernel("badOperand", params(declared)) { _ =>
      val ticket = atomicFetchAdd("ticket", missing(literal(0)), literal(1))
      declared(literal(0)) := ticket
    }
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector(ValidationCode.UnknownBuffer))

  test("capture effects retain target RMW address dependencies and local initialization"):
    val indices = input[Int]("indices")
    val definition = kernel("effects", params(indices)) { _ =>
      val data = sharedArray[Int]("data", 2)
      atomicFetchAdd("ticket", data(indices(literal(0)).read), literal(1))
      ()
    }
    val effects = EffectAnalysis.statement(definition.ir.body.statements.head)
    assertEquals(effects.readSpaces, Set(EffectMemorySpace.Shared, EffectMemorySpace.Global))
    assertEquals(effects.writtenSpaces, Set(EffectMemorySpace.Shared, EffectMemorySpace.Local))
    assert(!effects.hasBarrier)

  test("capture remains varying under a uniform address and triggers dependent barrier warnings"):
    val data = inOut[Int]("data")
    val definition = kernel("varyingResult", params(data)) { _ =>
      val ticket = atomicFetchAdd("ticket", data(literal(0)), literal(1))
      when(ticket < literal(16)) { barrier() }
    }
    val state = UniformityAnalysis.scopeAfter(definition.ir.body.statements.head, UniformityScope.empty)
    assertEquals(state.locals("ticket"), Uniformity.Varying)
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("optimization preserves unused and zero captures without folding or duplicating their results"):
    val data = inOut[Int]("data")
    val definition = kernel("preservation", params(data)) { _ =>
      atomicFetchAdd("unused", data(literal(0)), literal(0))
      val ticket = atomicFetchAdd("ticket", data(literal(0)), literal(1) + literal(1))
      data(literal(1)) := ticket + ticket
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.size, 3)
    assertEquals(normalized.body.statements.head, definition.ir.body.statements.head)
    assertEquals(normalized.body.statements(1).asInstanceOf[AtomicFetchAdd[Int, Global]].value, literal(2))
    assertEquals(normalized.body.statements.last, definition.ir.body.statements.last)
    assertEquals(IrNormalizer.kernel(normalized), normalized)
    assert(KernelValidator.validate(normalized).isValid)

  test("CSE reserves capture names and rewrites only pure operands"):
    val n = value[Int]("n")
    val data = inOut[Int]("data")
    val definition = kernel("captureCse", params(n, data)) { _ =>
      atomicFetchAdd("flight4s_cse_0", data(n * literal(2)), n * literal(2))
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    val temporary = normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].local
    assertEquals(temporary.name, "flight4s_cse_1")
    val capture = normalized.body.statements.last.asInstanceOf[AtomicFetchAdd[Int, Global]]
    assertEquals(capture.target, data(temporary.read))
    assertEquals(capture.value, temporary.read)
    assert(KernelValidator.validate(normalized).isValid)

  test("atomic old-value capture cannot be staged inside pure expression callbacks"):
    val data = inOut[Int]("data")
    val failure = intercept[DslError] {
      kernel("badCallback", params(data)) { _ =>
        gpuRange("i", literal(0), literal(2)).map { _ =>
          atomicFetchAdd("ticket", data(literal(0)), literal(1))
        }.foreach { value => data(literal(1)) := value }
      }
    }
    assertEquals(failure.code, DslErrorCode.StatementInsideExpression)

  test("atomicFetchAdd supports UInt and Float shared targets"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("sharedTickets") {
        val uints = sharedArray[UInt]("uints", 1)
        val floats = sharedArray[Float]("floats", 1)
        val u = atomicFetchAdd("u", uints(literal(0)), literal(UInt.fromBits(1)))
        val f = atomicFetchAdd("f", floats(literal(0)), literal(0.5f))
      }
    """), Nil)
