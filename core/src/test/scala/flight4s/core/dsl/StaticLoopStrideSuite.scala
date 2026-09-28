package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.{CodegenError, CudaCodegen}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.I32

class StaticLoopStrideSuite extends FunSuite:
  private val span = SourceSpan("Stride.scala", 12, 3, 12, 60)
  private given DslSourcePosition = DslSourcePosition(span)

  test("gpuFor accepts an explicit positive static step"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("stride", params(output[Int]("out"))) { p =>
        gpuFor("i", literal(0), literal(16), step = 4) { i => p._1(i) := i }
      }
    """), Nil)

  test("step is static Int metadata rather than a device expression"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("stride") { gpuFor("i", literal(0), literal(16), literal(4)) { _ => () } }
    """).nonEmpty)

  test("default and explicit unit strides retain identical IR and generated source"):
    val out = output[Int]("out")
    val signature = params(out)
    val old = kernel("unit", signature) { _ =>
      gpuFor("i", literal(0), literal(8)) { i => out(i) := i }
    }
    val explicit = kernel("unit", signature) { _ =>
      gpuFor("i", literal(0), literal(8), step = 1) { i => out(i) := i }
    }
    val loop = old.body.statements.head.asInstanceOf[ForLoop]
    assertEquals(loop, ForLoop(LoopIndex("i", span), literal(0), literal(8), loop.body, span))
    assertEquals(old.ir, explicit.ir)
    val generated = CudaCodegen.generate(old).toOption.get
    assertEquals(generated, CudaCodegen.generate(explicit).toOption.get)
    assertEquals(generated.cudaSource,
      "extern \"C\" __global__ void unit(int* out) {\n" +
      "  for (int i = 0; i < 8; ++i) {\n    out[i] = i;\n  }\n}\n")

  test("nonpositive steps produce staged validation errors before codegen"):
    Vector(0, -1, Int.MinValue).foreach { step =>
      val definition = kernel("invalid") {
        gpuFor("i", literal(0), literal(8), step) { _ => () }
      }
      val errors = KernelValidator.validate(definition).errors
      assertEquals(errors.map(_.code), Vector(ValidationCode.InvalidLoopStep))
      assertEquals(errors.head.location, "body.statements[0].step")
      assertEquals(errors.head.span, span)
      val moduleErrors = errors.map(error => error.copy(location = s"kernels[0].${error.location}"))
      assertEquals(CudaCodegen.generate(definition), Left(CodegenError.ValidationFailed(moduleErrors)))
    }

  test("a stride stages its body once even for empty or reversed ranges"):
    Vector((0, 8), (8, 8), (8, 0)).foreach { (from, until) =>
      var calls = 0
      val definition = kernel("staging") {
        gpuFor("i", literal(from), literal(until), 3) { i =>
          calls += 1
          assertEquals(i.valueType, I32)
        }
      }
      assertEquals(calls, 1)
      assertEquals(definition.body.statements.head.asInstanceOf[ForLoop].step, 3)
      assert(KernelValidator.validate(definition).isValid)
    }

  test("wide induction preserves Int callback indices source spans and collision-free names"):
    val out = output[Int]("out")
    val definition = kernel("wide", params(out, value[Int]("flight4s_value_0"))) { _ =>
      gpuFor("i", literal(0), literal(8), 3) { i =>
        gpuFor("j", literal(0), literal(4), 2) { j => out(i + j) := i }
      }
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("for (long long flight4s_value_1 = 0; flight4s_value_1 < 8; flight4s_value_1 += 3LL)"))
    assert(generated.cudaSource.contains("const int i = static_cast<int>(flight4s_value_1);"))
    assert(generated.cudaSource.contains("for (long long flight4s_value_2 = 0; flight4s_value_2 < 4; flight4s_value_2 += 2LL)"))
    generated.cudaSource.linesIterator.zipWithIndex.filter { (line, _) =>
      line.contains("for (") || line.contains("const int")
    }.foreach { (_, index) =>
      assertEquals(generated.sourceMap.closestAtOrBefore(index + 1).get.sourceSpan, span)
    }
    assertEquals(CudaCodegen.generate(definition).toOption.get, generated)

  test("normalization and CSE preserve stride and rechecked mutable upper bounds"):
    val definition = kernel("passes", params(value[Int]("n"))) { p =>
      val limit = local("limit", literal(8))
      gpuFor("i", p._1 + literal(1), limit.read, 3) { _ => limit := literal(0) }
    }
    val original = definition.body.statements(1).asInstanceOf[ForLoop]
    val normalized = IrNormalizer.kernel(definition.ir)
    val optimized = LocalCommonSubexpressionElimination.kernel(normalized)
    val loop = optimized.body.statements(1).asInstanceOf[ForLoop]
    assertEquals(loop.step, original.step)
    assertEquals(loop.until, original.until)
    assert(KernelValidator.validate(optimized).isValid)

  test("stride does not change memory effects or barrier uniformity classifications"):
    val definition = kernel("effects", params(output[Int]("out"))) { p =>
      gpuFor("i", literal(0), threadIdx.x, 3) { i =>
        p._1(i) := i
        barrier()
      }
    }
    val loop = definition.body.statements.head.asInstanceOf[ForLoop]
    val unit = loop.copy(step = 1)
    assertEquals(EffectAnalysis.statement(loop), EffectAnalysis.statement(unit))
    assertEquals(UniformityAnalysis.loopScope(loop, UniformityScope.empty),
      UniformityAnalysis.loopScope(unit, UniformityScope.empty))
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge))

  test("indices remain lexical and cannot conflict with parameters"):
    val out = output[Int]("out")
    val index = LoopIndex("i", span)
    val loop = ForLoop(index, literal(0), literal(8), Block(Vector.empty), span, 3)
    val escaped = KernelIR("escaped", params(out), Block(Vector(loop,
      Store(BufferElement[Int, ReadWrite]("out", literal(0), I32), index))))
    assertEquals(KernelValidator.validate(escaped).errors.map(_.code), Vector(ValidationCode.UnboundLoopIndex))
    val conflict = KernelIR("conflict", params(value[Int]("i")), Block(Vector(loop)))
    assertEquals(KernelValidator.validate(conflict).errors.map(_.code), Vector(ValidationCode.LoopIndexConflictsWithBinding))
