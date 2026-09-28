package flight4s.core.ir

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock}
import flight4s.core.types.*

class BlockDimensionSpecializationSuite extends FunSuite:
  test("required dimensions become I32 literals with original spans"):
    val span = SourceSpan("Shape.scala", 5, 7, 5, 17)
    val shape = LaunchBlock.xyz(8, 4, 2)
    Vector("x" -> 8, "y" -> 4, "z" -> 2).foreach { (axis, expected) =>
      val definition = kernel("dimension", params(output[Int]("out"))) { p =>
        p._1(literal(0)) := Intrinsic(s"blockDim.$axis", I32, span)
      }.requiringBlock(shape)
      val normalized = IrNormalizer.kernel(definition.ir)
      val store = normalized.body.statements.head.asInstanceOf[Store[Int, Global]]
      assertEquals(store.value, Literal(expected, I32, span))
      assertEquals(normalized.requiredBlock, Some(shape))
      assert(normalized.signature eq definition.signature)
      assertEquals(IrNormalizer.kernel(normalized), normalized)
      assertEquals(KernelValidator.validate(normalized).errors, Vector.empty)
    }

  test("unconstrained and standalone normalization cannot assume launch geometry"):
    val definition = kernel("dynamic", params(output[Int]("out"))) { p =>
      p._1(literal(0)) := blockDim.x
    }
    assertEquals(IrNormalizer.kernel(definition.ir), definition.ir)
    assertEquals(IrNormalizer.block(definition.body), definition.body)
    assertEquals(IrNormalizer.expression(blockDim.x), blockDim.x)
    assertEquals(IrNormalizer.statement(definition.body.statements.head), definition.body.statements)
    val generated = CudaCodegen.generate(definition).toOption.get
    assertEquals(generated.cudaSource,
      "extern \"C\" __global__ void dynamic(int* out) {\n  out[0] = static_cast<int>(blockDim.x);\n}\n")

  test("only known I32 block dimensions specialize"):
    val dynamic = Vector[Expr[?]](threadIdx.x, threadIdx.y, threadIdx.z,
      blockIdx.x, Intrinsic("gridDim.x", I32), Intrinsic("blockDim.w", I32), Intrinsic("blockDim.x", F32))
    dynamic.foreach { expression =>
      val local = LocalVariable("value", expression.valueType)
      val statement = LocalDeclaration(local, expression)
      val definition = KernelIR("other", params(), Block(Vector(statement)), requiredBlock = Some(LaunchBlock.x(32)))
      assertEquals(IrNormalizer.kernel(definition).body, definition.body)
    }

  test("known dimensions fold integer arithmetic and select branches without changing floating arithmetic"):
    val definition = kernel("select", params(output[Int]("out"), output[Float]("floats"))) { p =>
      val size = local("size", blockDim.x * blockDim.y * blockDim.z)
      gpuIf(size.read === literal(64)) {
        p._1(literal(0)) := choose(blockDim.z > literal(1))(blockDim.x + literal(2))(literal(-1))
      } {
        p._1(literal(0)) := literal(-2)
      }
      p._2(literal(0)) := literal(1.5f) + literal(2.5f)
    }.requiringBlock(LaunchBlock.xyz(8, 4, 2))
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]].initial, Literal(64, I32))
    val selected = normalized.body.statements(1).asInstanceOf[ScopedBlock]
    val originalChoice = definition.body.statements(1).asInstanceOf[IfThen]
      .thenBlock.statements.head.asInstanceOf[Store[Int, Global]].value
    assertEquals(selected.body.statements.head.asInstanceOf[Store[Int, Global]].value, Literal(10, I32, originalChoice.span))
    assert(normalized.body.statements.last.asInstanceOf[Store[Float, Global]].value.isInstanceOf[Binary[?]])
    assertEquals(KernelValidator.validate(normalized).errors, Vector.empty)

  test("dimensions survive branches, scopes, and loops while mutable local facts are cleared"):
    val definition = kernel("control", params(output[Int]("out"))) { p =>
      val bound = local("bound", literal(4))
      when(threadIdx.x === literal(0)) { bound := literal(2) }
      p._1(literal(0)) := bound.read + blockDim.x
      scoped { p._1(literal(1)) := blockDim.y }
      p._1(literal(2)) := blockDim.z
      gpuFor("i", literal(0), bound.read + blockDim.x) { i =>
        bound := literal(0)
        gpuFor("j", literal(0), blockDim.y) { j =>
          p._1(i + j) := blockDim.z
        }
      }
      p._1(literal(3)) := bound.read + blockDim.x
    }.requiringBlock(LaunchBlock.xyz(8, 4, 2))
    val normalized = IrNormalizer.kernel(definition.ir)
    val statements = normalized.body.statements
    def assertLiveBound(statement: Stmt): Unit =
      val value = statement.asInstanceOf[Store[Int, Global]].value.asInstanceOf[Binary[Int]]
      assert(value.left.isInstanceOf[Load[?, ?, ?]])
      assertEquals(value.right, Literal(8, I32))
    assertLiveBound(statements(2))
    assertLiveBound(statements.last)
    assertEquals(statements(3).asInstanceOf[ScopedBlock].body.statements.head.asInstanceOf[Store[Int, Global]].value, Literal(4, I32))
    assertEquals(statements(4).asInstanceOf[Store[Int, Global]].value, Literal(2, I32))
    val loop = statements(5).asInstanceOf[ForLoop]
    assert(loop.until.asInstanceOf[Binary[Int]].left.isInstanceOf[Load[?, ?, ?]])
    assertEquals(loop.until.asInstanceOf[Binary[Int]].right, Literal(8, I32))
    val inner = loop.body.statements(1).asInstanceOf[ForLoop]
    assertEquals(inner.until, Literal(4, I32))
    assertEquals(inner.body.statements.head.asInstanceOf[Store[Int, Global]].value, Literal(2, I32))
    assertEquals(IrNormalizer.kernel(normalized), normalized)

  test("functional folds and sums specialize and retain strict reduction policy"):
    val definition = kernel("functional", params(output[Int]("out"))) { p =>
      val values = gpuRange("i", literal(0), blockDim.x).map(i => i + blockDim.y)
      val sum = values.sum(literal(0))
      val folded = gpuRange("j", literal(0), blockDim.z).foldLeft("total", literal(0)) { (acc, j) =>
        acc + j + blockDim.x
      }
      p._1(literal(0)) := sum + folded
    }.requiringBlock(LaunchBlock.xyz(8, 4, 2))
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(!generated.cudaSource.contains("blockDim."), generated.cudaSource)
    assert(generated.cudaSource.contains("strict/serial-left-fold"))
    assertEquals(generated.launchRequirements.requiredBlock, definition.requiredBlock)
    assertEquals(KernelValidator.validate(IrNormalizer.kernel(definition.ir)).errors, Vector.empty)

  test("module specialization isolates each kernel's geometry"):
    def definition(name: String) = kernel(name, params(output[Int]("out"))) { p =>
      p._1(literal(0)) := blockDim.x * blockDim.y * blockDim.z
    }
    val first = definition("first").requiringBlock(LaunchBlock.x(32))
    val second = definition("second").requiringBlock(LaunchBlock.xy(8, 8))
    val dynamic = definition("dynamic")
    val normalized = IrNormalizer.module(CudaModuleIR(Vector.empty, Vector(first.ir, second.ir, dynamic.ir)))
    assertEquals(normalized.kernels(0).body.statements.head.asInstanceOf[Store[Int, Global]].value, Literal(32, I32))
    assertEquals(normalized.kernels(1).body.statements.head.asInstanceOf[Store[Int, Global]].value, Literal(64, I32))
    assertEquals(normalized.kernels(2), IrNormalizer.kernel(dynamic.ir))
    assertEquals(IrNormalizer.module(normalized), normalized)
