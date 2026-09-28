package flight4s.examples

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class BlockRowSoftmaxSuite extends FunSuite:
  test("a separate block-cooperative softmax exposes a complete runnable entry point"):
    assertEquals(typeCheckErrors("""
      import flight4s.examples.BlockRowSoftmax
      val output: Array[Float] = BlockRowSoftmax.run(Array.emptyFloatArray, 0, 3)
      BlockRowSoftmax.main(Array("--cuda-source"))
    """), Nil)

  test("host shape and finite-input contracts are enforced before opening CUDA"):
    intercept[IllegalArgumentException](BlockRowSoftmax.run(Array.emptyFloatArray, -1, 3))
    intercept[IllegalArgumentException](BlockRowSoftmax.run(Array.emptyFloatArray, 1, 0))
    intercept[IllegalArgumentException](BlockRowSoftmax.run(Array(1.0f), 1, 2))
    intercept[IllegalArgumentException](BlockRowSoftmax.run(Array.emptyFloatArray, 50000, 50000))
    Vector(Float.NaN, Float.PositiveInfinity, Float.NegativeInfinity).foreach { value =>
      intercept[IllegalArgumentException](BlockRowSoftmax.run(Array(value), 1, 1))
    }
    assertEquals(BlockRowSoftmax.run(Array.emptyFloatArray, 0, 3).toVector, Vector.empty)

  test("launch geometry is exactly one full fixed-size block per row"):
    assertEquals(BlockRowSoftmax.definition.requiredBlock, Some(LaunchBlock.x(128)))
    assertEquals(BlockRowSoftmax.generated.launchRequirements.requiredBlock, Some(LaunchBlock.x(128)))
    Vector(1, 3, 129, Int.MaxValue).foreach { rows =>
      assertEquals(BlockRowSoftmax.launchConfig(rows), LaunchConfig(Grid.x(rows), LaunchBlock.x(128)))
    }

  test("cooperative kernel validates with no divergent barrier warnings"):
    val validation = KernelValidator.validate(BlockRowSoftmax.definition)
    assertEquals(validation.errors, Vector.empty)
    assertEquals(validation.warnings, Vector.empty)
    val source = BlockRowSoftmax.generated.cudaSource
    assert(source.contains("__shared__ float scratch[128];"))
    assert(source.contains("+= 128LL"))
    assert(source.contains("::expf("))
    assertEquals(source.sliding("__syncthreads();".length).count(_ == "__syncthreads();"), 17)
    assertEquals(BlockRowSoftmax.generated.compilerOptions.additionalNvrtcOptions, Vector.empty)

  test("all reduction barriers remain outside lane-varying branches and protect scratch reuse"):
    val rowBody = BlockRowSoftmax.definition.body.statements.collectFirst { case branch: IfThen => branch.thenBlock }.get
    val barriers = rowBody.statements.collect { case barrier: Barrier => barrier }
    assertEquals(barriers.size, 17)
    val laneBranches = rowBody.statements.collect { case branch: IfThen => branch }
    assertEquals(laneBranches.size, 14)
    assert(laneBranches.forall(_.thenBlock.statements.forall(_.isInstanceOf[Store[?, ?]])))
    val maximumIndex = rowBody.statements.indexWhere {
      case declaration: LocalDeclaration[?] => declaration.local.name == "maximum"
      case _ => false
    }
    assert(maximumIndex >= 0)
    assert(rowBody.statements(maximumIndex + 1).isInstanceOf[Barrier])
    assert(rowBody.statements(maximumIndex + 2).isInstanceOf[Store[?, ?]])
