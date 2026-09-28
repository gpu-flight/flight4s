package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors

import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class GpuRangeSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("mapped ranges lower to the existing reduction without intermediate storage"):
    val source = input[Float]("source")
    def square(value: Expr[Float]): Expr[Float] = value * value
    val actual = gpuRange("i", literal(0), literal(16))
      .map(i => source(i).read)
      .map(square)
      .sum(literal(0.0f))
    val expected = reduceSum("i", literal(0), literal(16), literal(0.0f)) { i =>
      square(source(i).read)
    }
    assertEquals(actual, expected)
    val out = output[Float]("out")
    val signature = params(source, out)
    val composed = kernel("squaredSum", signature) { _ => out(literal(0)) := actual }
    val direct = kernel("squaredSum", signature) { _ => out(literal(0)) := expected }
    assertEquals(CudaCodegen.generate(composed), CudaCodegen.generate(direct))

  test("range foreach lowers to the existing GPU loop and preserves statement order"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("rangeLoop", signature) { _ =>
      gpuRange("i", literal(0), literal(16)).foreach { i =>
        out(i) := i + literal(1)
        barrier()
      }
    }
    val expected = kernel("rangeLoop", signature) { _ =>
      gpuFor("i", literal(0), literal(16)) { i =>
        out(i) := i + literal(1)
        barrier()
      }
    }
    assertEquals(actual.ir, expected.ir)
    assert(KernelValidator.validate(actual).isValid)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("mapping composes lazy host builders and stages one body per terminal"):
    var reads = 0
    var transforms = 0
    val source = input[Float]("source")
    val mapped = gpuRange("i", literal(0), literal(1000000))
      .map { i =>
        reads += 1
        source(i).read
      }
      .map { value =>
        transforms += 1
        value * literal(2.0f)
      }
    assertEquals((reads, transforms), (0, 0))
    val first = mapped.sum(literal(0.0f))
    assertEquals((reads, transforms), (1, 1))
    val second = mapped.sum(literal(0.0f))
    assertEquals((reads, transforms), (2, 2))
    assertEquals(first, second)

  test("mapped foreach retains lexical loop scope and explicit effects"):
    val source = input[Float]("source")
    val signature = params(source)
    val actual = kernel("mappedLoop", signature) { _ =>
      val acc = local("acc", literal(0.0f))
      gpuRange("i", literal(0), literal(8)).map(i => source(i).read).foreach { value =>
        accumulate(acc, value)
      }
    }
    val expected = kernel("mappedLoop", signature) { _ =>
      val acc = local("acc", literal(0.0f))
      gpuFor("i", literal(0), literal(8)) { i => accumulate(acc, source(i).read) }
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("range sums retain low-precision accumulation and every explicit policy"):
    val source = input[Float16]("source")
    for policy <- ReductionPolicy.values do
      val actual: Expr[Float] = gpuRange("i", literal(0), literal(8))
        .map(i => source(i).read).sum(literal(1.0f), policy)
      val expected = reduceSum("i", literal(0), literal(8), literal(1.0f), policy)(i => source(i).read)
      assertEquals(actual, expected)
      assertEquals(actual.valueType, F32)

  test("nested range sums compose with distinct scoped indices"):
    val source = input[Float]("source")
    val out = output[Float]("out")
    val total = gpuRange("row", literal(0), literal(2)).map { row =>
      gpuRange("column", literal(0), literal(4))
        .map(column => source(row * literal(4) + column).read)
        .sum(literal(0.0f))
    }.sum(literal(0.0f))
    val definition = kernel("nestedRanges", params(source, out)) { _ =>
      out(literal(0)) := total
    }
    assert(KernelValidator.validate(definition).isValid)
    assert(CudaCodegen.generate(definition).isRight)

  test("range wrappers preserve scope errors and divergence diagnostics"):
    val source = input[Float]("i")
    val out = output[Float]("out")
    val invalid = kernel("indexConflict", params(source, out)) { _ =>
      out(literal(0)) := gpuRange("i", literal(0), literal(8))
        .map(i => source(i).read).sum(literal(0.0f))
    }
    assert(KernelValidator.validate(invalid).errors.exists(
      _.code == ValidationCode.ReductionIndexConflictsWithParameter
    ))
    val divergent = kernel("divergentRange") {
      gpuRange("i", literal(0), threadIdx.x).foreach(_ => barrier())
    }
    assert(KernelValidator.validate(divergent).warnings.nonEmpty)

  test("range terminals forward their call-site source position"):
    val span = SourceSpan("RangeUsage.scala", 12, 3, 12, 60)
    val total = gpuRange("i", literal(0), literal(8)).map(i => i)
      .sum(literal(0))(using summon[AccumulatorType[Int, Int]], summon[AdditiveType[Int]], DslSourcePosition(span))
    assertEquals(total.span, span)
    val definition = kernel("positionedRange") {
      gpuRange("i", literal(0), literal(8)).foreach(_ => barrier())(
        using summon[BlockBuilder], DslSourcePosition(span)
      )
    }
    assertEquals(definition.body.statements.head.span, span)

  test("range transformations require staged expressions and valid accumulator types"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(8)).map(_ => 1)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(8)).map(_ => literal(true)).sum(literal(false))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.*
      val source = input[Float16]("source")
      gpuRange("i", literal(0), literal(8)).map(i => source(i).read)
        .sum(literal(Float16.fromBits(0.toShort)))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(8)).foreach(_ => ())
    """).nonEmpty)
