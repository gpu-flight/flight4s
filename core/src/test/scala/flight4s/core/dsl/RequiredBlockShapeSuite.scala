package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.codegen.{CudaCodegen, KernelLaunchRequirements}
import flight4s.core.ir.{IrNormalizer, LocalCommonSubexpressionElimination}
import flight4s.core.launch.{Block as LaunchBlock}

class RequiredBlockShapeSuite extends FunSuite:
  test("kernels can declare an exact required block shape"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.launch.{Block as LaunchBlock}
      val definition = kernel("cooperative") { () }.requiringBlock(LaunchBlock.x(128))
    """), Nil)

  test("block requirements are immutable and retain the original typed signature"):
    val original = kernel("cooperative") { () }
    val required = original.requiringBlock(LaunchBlock.x(128))
    assertEquals(original.requiredBlock, None)
    assertEquals(required.requiredBlock, Some(LaunchBlock.x(128)))
    assert(required.signature eq original.signature)
    assertEquals(required.body, original.body)
    assertEquals(required.requiringBlock(LaunchBlock.xy(8, 16)).requiredBlock, Some(LaunchBlock.xy(8, 16)))

  test("codegen carries exact dimensions without changing CUDA or existing shared requirements"):
    val original = kernel("cooperative") { dynamicSharedArray[Float]("scratch"); () }
    val plain = CudaCodegen.generate(original).toOption.get
    Vector(LaunchBlock.x(128), LaunchBlock.xy(8, 16), LaunchBlock.xyz(8, 4, 4)).foreach { shape =>
      val definition = original.requiringBlock(shape)
      val generated = CudaCodegen.generate(definition).toOption.get
      assertEquals(generated.cudaSource, plain.cudaSource)
      assertEquals(generated.sourceMap, plain.sourceMap)
      assertEquals(generated.launchRequirements, plain.launchRequirements.copy(requiredBlock = Some(shape)))
      val normalized = IrNormalizer.kernel(definition.ir)
      assertEquals(normalized.requiredBlock, Some(shape))
      assertEquals(LocalCommonSubexpressionElimination.kernel(normalized).requiredBlock, Some(shape))
    }

  test("unconstrained kernels remain unconstrained"):
    val generated = CudaCodegen.generate(kernel("plain") { () }).toOption.get
    assertEquals(generated.launchRequirements, KernelLaunchRequirements())
