package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class GpuRangeStrideSuite extends FunSuite:
  private val span = SourceSpan("RangeStride.scala", 9, 1, 9, 80)
  private given DslSourcePosition = DslSourcePosition(span)

  test("functional ranges accept a static stride before mapping and reduction"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("strided", params(output[Int]("out"))) { p =>
        p._1(literal(0)) := gpuRange("i", literal(0), literal(16)).by(4)
          .map(i => i * i).sum(literal(0))
      }
    """), Nil)

  test("by is immutable and replaces rather than multiplies the previous stride"):
    val original = gpuRange("i", literal(0), literal(16))
    val two = original.by(2)
    val three = two.by(3)
    assertEquals((original.step, two.step, three.step), (1, 2, 3))
    assertEquals((three.indexName, three.from, three.until), (original.indexName, original.from, original.until))
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(16)).by(literal(2))
    """).nonEmpty)

  test("mapped filtered traversal matches explicit strided loops exactly"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("mapped", signature) { _ =>
      gpuRange("i", literal(0), literal(16)).by(3).map(_ * literal(2))
        .filter(_ < literal(10)).map(_ + literal(1)).foreach(v => out(v) := v)
    }
    val expected = kernel("mapped", signature) { _ =>
      gpuFor("i", literal(0), literal(16), 3) { i =>
        val v = i * literal(2)
        when(v < literal(10)) { out(v + literal(1)) := v + literal(1) }
      }
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("scalar pair and filtered folds retain the selected stride"):
    val definition = kernel("folds") {
      val range = gpuRange("i", literal(0), literal(16)).by(3)
      range.foldLeft("scalar", literal(0))(_ + _)
      range.map(_ * literal(2)).foldLeft("pair", (literal(0), literal(0))) { (state, i) =>
        (state._1 + i, state._2 + literal(1))
      }
      range.filter(_ < literal(8)).foldLeft("filtered", literal(1))(_ + _)
      ()
    }
    val loops = definition.body.statements.collect { case loop: ForLoop => loop }
    assertEquals(loops.map(_.step), Vector(3, 3, 3))
    assert(KernelValidator.validate(definition).isValid)

  test("for generators retain independent strides and ordered guards"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("nested", signature) { _ =>
      val pairs = for
        i <- gpuRange("i", literal(0), literal(9)).by(3)
        j <- gpuRange("j", literal(0), i).by(2)
        if j < literal(4)
      yield i + j
      pairs.foreach(v => out(literal(0)) := v)
    }
    val expected = kernel("nested", signature) { _ =>
      gpuFor("i", literal(0), literal(9), 3) { i =>
        gpuFor("j", literal(0), i, 2) { j =>
          when(j < literal(4)) { out(literal(0)) := i + j }
        }
      }
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("strided sum retains accumulator types policies and direct reduction equivalence"):
    val source = input[Float16]("source")
    val out = output[Float]("out")
    for policy <- ReductionPolicy.values do
      val actual = gpuRange("i", literal(0), literal(16)).by(3)
        .map(i => source(i).read).sum(literal(1.0f), policy)
      val expected = reduceSum("i", literal(0), literal(16), literal(1.0f), policy, step = 3)(i => source(i).read)
      assertEquals(actual, expected)
      assertEquals(actual.valueType, F32)
      val reduction = actual.asInstanceOf[ReduceSum[Float16, Float]]
      assertEquals(IrNormalizer.expression(actual).asInstanceOf[ReduceSum[Float16, Float]].step, 3)
      assertEquals(EffectAnalysis.expression(reduction), EffectAnalysis.expression(reduction.copy(step = 1)))
      assertEquals(UniformityAnalysis.expression(reduction), UniformityAnalysis.expression(reduction.copy(step = 1)))
      val definition = kernel("half", params(source, out)) { _ => out(literal(0)) := actual }
      assert(KernelValidator.validate(definition).isValid)
      val generated = CudaCodegen.generate(definition).toOption.get
      assert(generated.cudaSource.contains("__half2float(source[i])"))
      assert(generated.cudaSource.contains("+= 3LL"))
      assertEquals(generated.sourceMap.entries.last.sourceSpan, span)

  test("unit-stride sums retain the same IR and exact CUDA source"):
    val out = output[Int]("out")
    val actual = gpuRange("i", literal(0), literal(4)).by(1).map(identity).sum(literal(0))
    val expected = reduceSum("i", literal(0), literal(4), literal(0))(identity)
    assertEquals(actual, expected)
    val definition = kernel("unit", params(out)) { _ => out(literal(0)) := actual }
    assertEquals(CudaCodegen.generate(definition).toOption.get.cudaSource,
      "extern \"C\" __global__ void unit(int* out) {\n" +
      "  out[0] = ([&]() { /* flight4s reduction: strict/serial-left-fold */ int flight4s_accumulator_0 = 0; " +
      "for (int i = 0; i < 4; ++i) { flight4s_accumulator_0 += i; } return flight4s_accumulator_0; }());\n}\n")

  test("invalid steps are diagnosed for every terminal with terminal source spans"):
    for step <- Vector(0, -1, Int.MinValue) do
      val definition = kernel("invalid", params(output[Int]("out"))) { p =>
        val range = gpuRange("i", literal(0), literal(4)).by(step)
        range.foreach(_ => ())
        range.foldLeft("total", literal(0))(_ + _)
        p._1(literal(0)) := range.map(identity).sum(literal(0))
      }
      val errors = KernelValidator.validate(definition).errors
      assertEquals(errors.map(_.code), Vector.fill(3)(ValidationCode.InvalidLoopStep))
      assert(errors.forall(_.span == span))
      assertEquals(errors.map(_.location), Vector("body.statements[0].step", "body.statements[2].step", "body.statements[3].value.step"))
      assert(CudaCodegen.generate(definition).isLeft)

  test("mapping stays lazy and stages once per terminal regardless of stride or empty bounds"):
    var calls = 0
    val range = gpuRange("i", literal(8), literal(0)).by(3).map { i => calls += 1; i }
    assertEquals(calls, 0)
    val a = range.sum(literal(1))
    val b = range.sum(literal(1))
    assertEquals(calls, 2)
    assertEquals(a, b)

  test("invalid reduction steps are rejected even inside an unreachable conditional arm"):
    val definition = kernel("invalid_arm", params(output[Int]("out"))) { p =>
      p._1(literal(0)) := choose(literal(false))(
        gpuRange("i", literal(0), literal(4)).by(0).map(identity).sum(literal(0))
      )(literal(7))
    }
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector(ValidationCode.InvalidLoopStep))
    assert(CudaCodegen.generate(definition).isLeft)

  test("strided expression callbacks still reject statement effects"):
    val error = intercept[DslError] {
      kernel("bad") {
        gpuRange("i", literal(0), literal(16)).by(3).map { i => barrier(); i }.sum(literal(0))
        ()
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)
