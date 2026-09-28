package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock}

class BlockReductionSuite extends FunSuite:
  test("block scratch can be reused by named sums and custom ordered trees"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.launch.{Block as LaunchBlock}
      kernel("reduce") {
        val reduction = block.reduction[Float]("scratch", LaunchBlock.x(128))
        val sum: Expr[Float] = reduction.sum("sum", literal(1.0f))
        val product: Expr[Float] = reduction.reduceTree("product", literal(2.0f))(_ * _)
      }
    """), Nil)

  test("every use records its exact shape and generated launch metadata enforces it"):
    val shape = LaunchBlock.xyz(4, 4, 4)
    val definition = kernel("contract") {
      val reduction = block.reduction[Int]("scratch", shape)
      gpuFor("i", literal(0), literal(2)) { _ => reduction.sum("sum", threadIdx.x); () }
    }
    assertEquals(definition.requiredBlock, Some(shape))
    assertEquals(definition.ir.blockRequirements.map(_.shape), Vector(shape))
    assertNotEquals(definition.ir.blockRequirements.head.span, SourceSpan.Unknown)
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    assertEquals(CudaCodegen.generate(definition).toOption.get.launchRequirements.requiredBlock, Some(shape))
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.blockRequirements, definition.ir.blockRequirements)
    assertEquals(LocalCommonSubexpressionElimination.kernel(normalized).blockRequirements, definition.ir.blockRequirements)

  test("overrides missing metadata and mixed shapes cannot invalidate collective contracts silently"):
    val definition = kernel("contract") {
      block.reduction[Int]("scratch", LaunchBlock.x(32)).sum("sum", literal(1))
      ()
    }
    for invalid <- Vector(definition.requiringBlock(LaunchBlock.x(64)).ir,
      definition.requiringBlock(LaunchBlock.xy(8, 4)).ir, definition.ir.copy(requiredBlock = None)) do
      assert(KernelValidator.validate(invalid).errors.exists(_.code == ValidationCode.ConflictingBlockRequirement))
      assert(CudaCodegen.generate(Kernel(invalid)).isLeft)
    val mixed = kernel("mixed") {
      block.reduction[Int]("a", LaunchBlock.x(32)).sum("first", literal(1))
      block.reduction[Int]("b", LaunchBlock.x(64)).sum("second", literal(1))
      ()
    }
    assert(KernelValidator.validate(mixed).errors.exists(_.code == ValidationCode.ConflictingBlockRequirement))

  test("shape arithmetic rejects non-powers and enormous dimensions without overflow"):
    for shape <- Vector(LaunchBlock.x(3), LaunchBlock.x(1025), LaunchBlock.xyz(Int.MaxValue, Int.MaxValue, Int.MaxValue)) do
      assertEquals(intercept[DslError] {
        kernel("bad") { block.reduction[Int]("scratch", shape); () }
      }.code, DslErrorCode.InvalidBlockReductionShape)
    for size <- Vector(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) do
      val definition = kernel("valid") { block.reduction[Int]("scratch", LaunchBlock.x(size)).sum("sum", literal(1)); () }
      assertEquals(KernelValidator.validate(definition).errors, Vector.empty)

  test("shared scratch has one allocation and every tree call ends with a reuse barrier"):
    var callbacks = 0
    val definition = kernel("reuse") {
      val reduction = block.reduction[Float]("scratch", LaunchBlock.x(8))
      for name <- Vector("first", "second") do
        reduction.reduceTree(name, literal(1.0f)) { (left, right) =>
          callbacks += 1
          left - right
        }
    }
    assertEquals(callbacks, 6)
    assertEquals(definition.sharedMemory.size, 1)
    val bodies = definition.body.statements.collect { case scope: ScopedBlock => scope.body }
    assertEquals(bodies.size, 2)
    for body <- bodies do
      assert(body.statements.last.isInstanceOf[Barrier])
      assert(body.statements(body.statements.size - 2).isInstanceOf[Store[?, ?]])
      assertEquals(body.statements.count(_.isInstanceOf[Barrier]), 5)
    assertEquals(KernelValidator.validate(definition).warnings, Vector.empty)

  test("scratch declarations and callbacks retain DSL effect and scope checks"):
    assertEquals(intercept[DslError] {
      kernel("nestedDeclaration") { scoped { block.reduction[Int]("scratch", LaunchBlock.x(2)); () } }
    }.code, DslErrorCode.SharedMemoryDeclarationOutsideKernelBody)
    assertEquals(intercept[DslError] {
      kernel("effectful") {
        block.reduction[Int]("scratch", LaunchBlock.x(2)).reduceTree("sum", literal(1)) { (left, right) =>
          barrier()
          left + right
        }
        ()
      }
    }.code, DslErrorCode.StatementInsideExpression)
    assertEquals(intercept[DslError] {
      kernel("hidden") {
        val reduction = block.reduction[Int]("scratch", LaunchBlock.x(1))
        let("bad", choose(literal(true))(reduction.sum("sum", literal(1)))(literal(0)))
        ()
      }
    }.code, DslErrorCode.StatementInsideExpression)

  test("divergent use remains diagnosed and unused scratch does not constrain the launch"):
    val unused = kernel("unused") { block.reduction[Int]("scratch", LaunchBlock.x(32)); () }
    assertEquals(unused.requiredBlock, None)
    val divergent = kernel("divergent") {
      val reduction = block.reduction[Int]("scratch", LaunchBlock.x(32))
      when(threadIdx.x < literal(16)) { reduction.sum("sum", literal(1)); () }
    }
    assertEquals(KernelValidator.validate(divergent).warnings.map(_.code),
      Vector.fill(7)(ValidationWarningCode.BarrierMayDiverge))

  test("scratch handles reused in another kernel still register a shape and require a declaration"):
    var captured: Option[BlockReduction[Int]] = None
    kernel("owner") { captured = Some(block.reduction[Int]("scratch", LaunchBlock.x(32))) }
    val other = kernel("other") { captured.get.sum("sum", literal(1)); () }
    assertEquals(other.requiredBlock, Some(LaunchBlock.x(32)))
    assert(KernelValidator.validate(other).errors.exists(_.code == ValidationCode.UnknownSharedMemory))

  test("two-thread lowering exactly matches explicit shared-memory statements"):
    given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
    val out = output[Int]("out")
    val signature = params(out)
    val generated = kernel("tree", signature) { _ =>
      val reduction = block.reduction[Int]("scratch", LaunchBlock.x(2))
      val result = reduction.reduceTree("result", threadIdx.x)(_ - _)
      out(threadIdx.x) := result
    }
    val explicit = kernel("tree", signature) { _ =>
      val scratch = sharedArray[Int]("scratch", 2)
      val result = local("result", threadIdx.x)
      scoped {
        val rank = let("result_rank", threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
        scratch(rank) := result.read
        barrier()
        when((rank % literal(2)) === literal(0)) {
          scratch(rank) := scratch(rank).read - scratch(rank + literal(1)).read
        }
        barrier()
        result := scratch(literal(0)).read
        barrier()
      }
      out(threadIdx.x) := result.read
    }.requiringBlock(LaunchBlock.x(2))
    assertEquals(generated.ir.copy(blockRequirements = Vector.empty), explicit.ir)
    assertEquals(CudaCodegen.generate(generated), CudaCodegen.generate(explicit))

  test("custom Boolean trees compile but unsupported sums and mismatched combines do not"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.launch.{Block as LaunchBlock}
      kernel("all") { block.reduction[Boolean]("scratch", LaunchBlock.x(32))
        .reduceTree("all", threadIdx.x < literal(16))(_ && _); () }
    """), Nil)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.launch.{Block as LaunchBlock}
      kernel("bad") { block.reduction[Boolean]("scratch", LaunchBlock.x(32)).sum("x", literal(true)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.launch.{Block as LaunchBlock}
      kernel("bad") { block.reduction[Int]("scratch", LaunchBlock.x(32))
        .reduceTree("x", literal(1))((a,b) => literal(1.0)); () }
    """).nonEmpty)
