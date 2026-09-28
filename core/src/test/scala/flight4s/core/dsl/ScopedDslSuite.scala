package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors

import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class ScopedDslSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("reusable helpers get separate sibling declaration scopes"):
    val out = output[Float]("out")
    def write(index: Int, value: Float)(using BlockBuilder): Unit = scoped {
      val temporary = local("temporary", literal(value))
      out(literal(index)) := temporary.read
    }
    val definition = kernel("repeatedHelper", params(out)) { _ =>
      write(0, 2.0f)
      write(1, 4.0f)
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    assert(definition.ir.body.statements.forall(_.isInstanceOf[ScopedBlock]))
    assert(CudaCodegen.generate(definition).isRight)

  test("scoped DSL lowers to the existing lexical block exactly"):
    val out = output[Float]("out")
    val signature = params(out)
    val body = kernel("helperBody", signature) { _ =>
      val temporary = local("temporary", literal(2.0f))
      out(literal(0)) := temporary.read
      barrier()
    }
    val definition = kernel("scopedHelper", signature) { _ => scoped {
      val temporary = local("temporary", literal(2.0f))
      out(literal(0)) := temporary.read
      barrier()
    } }
    val expected = Kernel(body.ir.copy(
      name = "scopedHelper", body = Block(Vector(ScopedBlock(body.ir.body)))
    ))
    assertEquals(definition.ir, expected.ir)
    assertEquals(CudaCodegen.generate(definition), CudaCodegen.generate(expected))

  test("locals and arrays cannot escape a scoped body"):
    var escaped = Option.empty[Expr[Float]]
    var escapedArray = Option.empty[Expr[Float]]
    val out = output[Float]("out")
    val definition = kernel("escapedLocal", params(out)) { _ =>
      scoped {
        escaped = Some(local("temporary", literal(2.0f)).read)
        val scratch = localArray[Float]("scratch", 2)
        scratch(literal(0)) := literal(3.0f)
        escapedArray = Some(scratch(literal(0)).read)
      }
      out(literal(0)) := escaped.get
      out(literal(1)) := escapedArray.get
    }
    assertEquals(
      KernelValidator.validate(definition).errors.map(_.code),
      Vector(ValidationCode.UnboundLocal, ValidationCode.UnknownLocalArray)
    )
    assert(CudaCodegen.generate(definition).isLeft)

  test("nested scopes access outer locals but cannot shadow active bindings"):
    val valid = kernel("outerAccess") {
      val counter = local("counter", literal(1))
      scoped { scoped { counter := counter.read + literal(2) } }
    }
    assert(KernelValidator.validate(valid).isValid)
    val invalid = kernel("outerShadow") {
      local("counter", literal(1))
      scoped { local("counter", literal(2)); () }
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code),
      Vector(ValidationCode.DuplicateLocalName))

  test("scopes compose with ranges and retain barrier divergence analysis"):
    val definition = kernel("scopedRanges") {
      scoped { gpuRange("i", literal(0), literal(4)).foreach(_ => barrier()) }
      scoped { gpuRange("i", literal(0), literal(4)).foreach(_ => barrier()) }
      scoped { when(threadIdx.x < literal(1)) { barrier() } }
    }
    val result = KernelValidator.validate(definition)
    assertEquals(result.errors, Vector.empty)
    assertEquals(result.warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))
    val effects = EffectAnalysis.block(definition.body)
    assert(effects.hasBarrier)

  test("scopes preserve shared-memory declaration ownership"):
    val error = intercept[DslError] {
      kernel("nestedShared") { scoped { sharedArray[Float]("scratch", 8); () } }
    }
    assertEquals(error.code, DslErrorCode.SharedMemoryDeclarationOutsideKernelBody)
    val definition = kernel("outerShared") {
      val scratch = sharedArray[Float]("scratch", 8)
      scoped { scratch(threadIdx.x) := literal(1.0f) }
    }
    assert(KernelValidator.validate(definition).isValid)
    assertEquals(definition.ir.sharedMemory.size, 1)

  test("scopes forward source positions and require a statement builder"):
    val span = SourceSpan("Helper.scala", 20, 2, 23, 3)
    val definition = kernel("positionedScope") {
      scoped { barrier() }(using summon[BlockBuilder], DslSourcePosition(span))
    }
    assertEquals(definition.body.statements.head.span, span)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      scoped { () }
    """).nonEmpty)
