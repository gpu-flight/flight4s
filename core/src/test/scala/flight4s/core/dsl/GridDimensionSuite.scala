package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock}
import flight4s.core.types.*

class GridDimensionSuite extends FunSuite:
  private val dimensions = Vector("x", "y", "z").map(axis => Intrinsic(s"gridDim.$axis", I32))

  test("gridDim exposes three typed Int dimensions"):
    val exposed: Vector[Expr[Int]] = Vector(gridDim.x, gridDim.y, gridDim.z)
    assertEquals(exposed, dimensions)
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      val dimensions: Vector[Expr[Int]] = Vector(gridDim.x, gridDim.y, gridDim.z)
    """), Nil)

  test("all grid dimensions validate and emit signed CUDA expressions"):
    dimensions.foreach { dimension =>
      val definition = kernel("gridSize", params(output[Int]("out"))) { p =>
        p._1(literal(0)) := dimension
      }
      assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      assertEquals(generated.cudaSource,
        s"extern \"C\" __global__ void gridSize(int* out) {\n  out[0] = static_cast<int>(${dimension.name});\n}\n")
    }

  test("grid dimensions are pure and grid-uniform but do not make thread expressions uniform"):
    dimensions.foreach { dimension =>
      assertEquals(EffectAnalysis.expression(dimension), EffectSummary.empty)
      assertEquals(UniformityAnalysis.expression(dimension), Uniformity.GridUniform)
      assertEquals(UniformityAnalysis.expression(dimension + blockIdx.x), Uniformity.BlockUniform)
      assertEquals(UniformityAnalysis.expression(dimension + threadIdx.x), Uniformity.Varying)
    }

  test("grid-dependent barriers are uniform and mixed thread-dependent barriers still warn"):
    val uniform = kernel("uniformGrid") {
      val size = local("size", dimensions.head)
      when(size.read > literal(1)) { barrier() }
      gpuRange("i", literal(0), dimensions(1)).foreach(_ => barrier())
    }
    assertEquals(KernelValidator.validate(uniform).errors, Vector.empty)
    assertEquals(KernelValidator.validate(uniform).warnings, Vector.empty)
    val varying = kernel("varyingGrid") {
      when(threadIdx.x < dimensions.head) { barrier() }
    }
    assertEquals(KernelValidator.validate(varying).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("malformed grid intrinsic types and unknown axes retain source-located diagnostics"):
    val span = SourceSpan("Grid.scala", 9, 4, 9, 14)
    val wrongType = kernel("wrongType", params(output[Float]("out"))) { p =>
      p._1(literal(0)) := Intrinsic("gridDim.x", F32, span)
    }
    val unknown = kernel("unknownAxis", params(output[Int]("out"))) { p =>
      p._1(literal(0)) := Intrinsic("gridDim.w", I32, span)
    }
    val typeErrors = KernelValidator.validate(wrongType).errors
    assertEquals(typeErrors.map(_.code), Vector(ValidationCode.ExpressionTypeMismatch))
    assertEquals(typeErrors.head.span, span)
    assertEquals(KernelValidator.validate(unknown).errors.map(_.code), Vector(ValidationCode.UnknownIntrinsic))
    assertEquals(KernelValidator.validate(unknown).errors.head.span, span)

  test("required block geometry does not specialize grid geometry"):
    dimensions.foreach { dimension =>
      val definition = kernel("stillDynamic", params(output[Int]("out"))) { p =>
        p._1(literal(0)) := dimension
      }.requiringBlock(LaunchBlock.xyz(8, 4, 2))
      assertEquals(IrNormalizer.kernel(definition.ir), definition.ir)
      assertEquals(IrNormalizer.expression(dimension), dimension)
    }

  test("functional ranges retain dynamic grid-dependent bounds and map expressions"):
    val definition = kernel("gridFold", params(output[Int]("out"))) { p =>
      p._1(literal(0)) := gpuRange("i", literal(0), dimensions(1))
        .map(i => i + dimensions(2)).sum(literal(0))
    }.requiringBlock(LaunchBlock.x(32))
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    assert(generated.cudaSource.contains("i < static_cast<int>(gridDim.y)"))
    assert(generated.cudaSource.contains("(i + static_cast<int>(gridDim.z))"))
    assert(generated.cudaSource.contains("strict/serial-left-fold"))
