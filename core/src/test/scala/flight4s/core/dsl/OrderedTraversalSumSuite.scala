package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class OrderedTraversalSumSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("named sums compose after filtered and nested scalar traversals"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      kernel("namedSums") {
        val filtered: Expr[Int] = gpuRange("i", literal(0), literal(8))
          .filter(_ > literal(0)).sum("filtered", literal(3))
        val nested: Expr[Float] = gpuRange("j", literal(0), literal(8))
          .flatMap(j => gpuRange("k", literal(0), j).map(k => convert.i32ToF32(k)))
          .sum("nested", literal(1.0f))
        ()
      }
    """), Nil)

  test("named sums preserve the complete IR and CUDA of explicit promoted ordered folds"):
    case class Entry(index: Expr[Int]) derives ProductFoldState
    val range = gpuRange("i", literal(0), literal(9)).by(2)
    val traversals = Vector[GpuTraversal[Int]](
      range, range.map(_ + literal(1)), range.filter(_ > literal(0)).map(i => literal(12) / i),
      range.flatMap(i => gpuRange("j", literal(0), i).map(j => i + j)),
      range.map(i => (i, i)).map(p => p._1 + p._2),
      range.map(i => Entry(i)).map(_.index)
    )
    val out = output[Int]("out")
    val signature = params(out)
    for traversal <- traversals do
      val actual = kernel("orderedSum", signature) { _ =>
        val sum = traversal.sum("sum", literal(3))
        out(literal(0)) := sum
        out(literal(1)) := sum
      }
      val expected = kernel("orderedSum", signature) { _ =>
        val sum = local("sum", literal(3))
        traversal.foreach(x => sum := sum.read + x.toAccumulator[Int])
        out(literal(0)) := sum.read
        out(literal(1)) := sum.read
      }
      assertEquals(KernelValidator.validate(actual).errors, Vector.empty)
      assertEquals(actual.ir, expected.ir)
      assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("low-precision elements use the existing explicit accumulator promotion rules"):
    def check[T](using CudaType[T], AccumulatorType[T, Float]): Unit =
      val source = input[T]("source")
      val out = output[Float]("out")
      val signature = params(source, out)
      val actual = kernel("promotedSum", signature) { _ =>
        out(literal(0)) := gpuRange("i", literal(0), literal(8)).filter(_ > literal(0))
          .map(i => source(i).read).sum("sum", literal(1.0f))
      }
      val expected = kernel("promotedSum", signature) { _ =>
        out(literal(0)) := gpuRange("i", literal(0), literal(8)).filter(_ > literal(0))
          .map(i => source(i).read).foldLeft("sum", literal(1.0f))((acc, x) => acc + x.toAccumulator[Float])
      }
      assertEquals(actual.ir, expected.ir)
      assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))
      assertEquals(KernelValidator.validate(actual).errors, Vector.empty)
    check[Float16]
    check[BFloat16]
    check[Float8E4M3]
    check[Float8E5M2]

  test("sum terminal stages each callback once even for empty ranges and repeated result reads"):
    var calls = 0
    val out = output[Int]("out")
    val definition = kernel("sumOnce", params(out)) { _ =>
      val sum = gpuRange("i", literal(0), literal(0)).map { i => calls += 1; i }
        .sum("sum", literal(7))
      out(literal(0)) := sum
      out(literal(1)) := sum
    }
    assertEquals(calls, 1)
    assertEquals(definition.body.statements.count(_.isInstanceOf[ForLoop]), 1)

  test("named sums require statement context and reject nested hidden statement emission"):
    val error = intercept[DslError] {
      kernel("hiddenSum") {
        gpuRange("i", literal(0), literal(4)).map { i =>
          gpuRange("j", literal(0), i).sum("hidden", literal(0))
        }.sum("outerSum", literal(0))
        ()
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)
    for callback <- Vector("map", "filter", "flatMap") do
      intercept[DslError] {
        kernel("badSumCallback") {
          gpuRange("i", literal(0), literal(4))
            .map { i => if callback == "map" then barrier(); i }
            .filter { i => if callback == "filter" then barrier(); i > literal(0) }
            .flatMap { i =>
              if callback == "flatMap" then barrier()
              gpuRange("j", literal(0), i)
            }.sum("sum", literal(0))
          ()
        }
      }
    assert(KernelValidator.validate(kernel("afterSum") { barrier() }).isValid)

  test("named sum results preserve lexical validation and cannot escape their scope"):
    var escaped = Option.empty[Expr[Int]]
    val out = output[Int]("out")
    val invalid = kernel("escapedSum", params(out)) { _ =>
      scoped { escaped = Some(gpuRange("i", literal(0), literal(4)).sum("sum", literal(0))) }
      out(literal(0)) := escaped.get
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val conflict = kernel("sumIndexConflict") {
      gpuRange("i", literal(0), literal(4)).sum("i", literal(0))
      ()
    }
    assertEquals(KernelValidator.validate(conflict).errors.map(_.code), Vector(ValidationCode.LoopIndexConflictsWithBinding))

  test("named sums forward source positions to declarations loops stores and results"):
    val span = SourceSpan("Sum.scala", 9, 2, 9, 80)
    var result = Option.empty[Expr[Int]]
    val definition = kernel("sumPosition") {
      result = Some(gpuRange("i", literal(0), literal(4)).sum("sum", literal(0))(
        using summon[AccumulatorType[Int, Int]], I32, summon[BlockBuilder], DslSourcePosition(span)))
    }
    assert(definition.body.statements.forall(_.span == span))
    assertEquals(definition.body.statements(1).asInstanceOf[ForLoop].body.statements.head.span, span)
    assertEquals(result.get.span, span)

  test("unsupported sum types policies mutability and expression-only contexts do not type-check"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(4)).sum("sum", literal(0))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { gpuRange("i", literal(0), literal(4)).map(_ => literal(true)).sum("sum", literal(false)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { gpuRange("i", literal(0), literal(4)).sum("sum", literal(0.0f)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.ReductionPolicy
      kernel("bad") { gpuRange("i", literal(0), literal(4)).sum("sum", literal(0), ReductionPolicy.Strict); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { gpuRange("i", literal(0), literal(4)).sum("sum", literal(0)) := literal(1) }
    """).nonEmpty)

  test("the existing expression-only sum retains every reduction policy and exact node structure"):
    val source = input[Float16]("source")
    for policy <- ReductionPolicy.values do
      val actual = gpuRange("i", literal(0), literal(8)).by(2)
        .map(i => source(i).read).sum(literal(1.0f), policy)
      val expected = reduceSum("i", literal(0), literal(8), literal(1.0f), policy, 2)(i => source(i).read)
      assertEquals(actual, expected)
