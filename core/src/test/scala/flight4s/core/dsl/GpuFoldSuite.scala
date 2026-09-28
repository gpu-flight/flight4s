package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors

import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class GpuFoldSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("range folds stage the existing local and serial loop exactly"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("serialFold", signature) { _ =>
      val folded = gpuRange("i", literal(0), literal(4))
        .foldLeft("acc", literal(2))((acc, i) => acc * literal(3) + i)
      out(literal(0)) := folded
      out(literal(1)) := folded
    }
    val expected = kernel("serialFold", signature) { _ =>
      val acc = local("acc", literal(2))
      gpuFor("i", literal(0), literal(4)) { i => acc := acc.read * literal(3) + i }
      out(literal(0)) := acc.read
      out(literal(1)) := acc.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("mapped fold callbacks stage once regardless of range length or result reads"):
    var mappings = 0
    var steps = 0
    val out = output[Int]("out")
    val definition = kernel("stagedFold", params(out)) { _ =>
      val range = gpuRange("i", literal(0), literal(1000000)).map { i => mappings += 1; i }
      assertEquals(mappings, 0)
      val folded = range.foldLeft("acc", literal(0)) { (acc, i) => steps += 1; acc + i }
      out(literal(0)) := folded
      out(literal(1)) := folded
    }
    assertEquals((mappings, steps), (1, 1))
    assertEquals(definition.body.statements.count(_.isInstanceOf[ForLoop]), 1)

  test("mapped folds allow scalar state types independent of the mapped element type"):
    val source = input[Float]("source")
    val out = output[Int]("out")
    val signature = params(source, out)
    val actual = kernel("countPositive", signature) { _ =>
      val count = gpuRange("i", literal(0), literal(8)).map(i => source(i).read)
        .foldLeft("count", literal(0)) { (count, x) =>
          choose(x > literal(0.0f))(count + literal(1))(count)
        }
      out(literal(0)) := count
    }
    val expected = kernel("countPositive", signature) { _ =>
      val count = local("count", literal(0))
      gpuFor("i", literal(0), literal(8)) { i =>
        count := choose(source(i).read > literal(0.0f))(count.read + literal(1))(count.read)
      }
      out(literal(0)) := count.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("fold state supports Boolean values and explicit low-precision conversion"):
    val source = input[Float16]("source")
    val out = output[Float]("out")
    val flags = output[Boolean]("flags")
    val definition = kernel("foldTypes", params(source, out, flags)) { _ =>
      val sum = gpuRange("i", literal(0), literal(8)).map(i => source(i).read)
        .foldLeft("sum", literal(0.0f))((acc, x) => acc + x.toAccumulator[Float])
      val all = gpuRange("j", literal(0), literal(8)).map(j => source(j).read)
        .foldLeft("all", literal(true)) { (acc, x) =>
          choose(acc)(x.toAccumulator[Float] > literal(0.0f))(literal(false))
        }
      out(literal(0)) := sum
      flags(literal(0)) := all
    }
    assert(KernelValidator.validate(definition).isValid)
    assert(CudaCodegen.generate(definition).isRight)

  test("fold declarations remain lexical and names must not conflict"):
    var escaped = Option.empty[Expr[Int]]
    val out = output[Int]("out")
    val invalid = kernel("escapedFold", params(out)) { _ =>
      scoped {
        escaped = Some(gpuRange("i", literal(0), literal(4)).foldLeft("acc", literal(0))(_ + _))
      }
      out(literal(0)) := escaped.get
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val conflict = kernel("foldIndexConflict") {
      gpuRange("i", literal(0), literal(4)).foldLeft("i", literal(0))(_ + _)
      ()
    }
    assertEquals(KernelValidator.validate(conflict).errors.map(_.code),
      Vector(ValidationCode.LoopIndexConflictsWithBinding))

  test("fold terminals forward source positions to declarations loops stores and results"):
    val span = SourceSpan("Fold.scala", 9, 2, 9, 80)
    var result = Option.empty[Expr[Int]]
    val definition = kernel("foldPosition") {
      result = Some(gpuRange("i", literal(0), literal(4)).foldLeft("acc", literal(0))(_ + _)(
        using I32, summon[BlockBuilder], DslSourcePosition(span)
      ))
    }
    assert(definition.body.statements.forall(_.span == span))
    assertEquals(definition.body.statements(1).asInstanceOf[ForLoop].body.statements.head.span, span)
    assertEquals(result.get.span, span)

  test("fold results are expressions and step state types must remain stable"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(4)).foldLeft("acc", literal(0))(_ + _)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("badFold") {
        gpuRange("i", literal(0), literal(4)).foldLeft("acc", literal(0))((_, _) => literal(1.0f))
      }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("badWrite") {
        val result = gpuRange("i", literal(0), literal(4)).foldLeft("acc", literal(0))(_ + _)
        result := literal(9)
      }
    """).nonEmpty)
