package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class FilteredGpuRangeSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("Scala for guards accept staged Boolean expressions"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("guardedFor") {
        for i <- gpuRange("i", literal(0), literal(8)) if i > literal(2) do
          barrier()
        val selected = for i <- gpuRange("j", literal(0), literal(8)) if i > literal(2)
          yield i * literal(2)
        selected.foreach(_ => ())
      }
    """), Nil)

  test("for guards lower to the same ordered loop and nested branches"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("guardOrder", signature) { _ =>
      for
        i <- gpuRange("i", literal(0), literal(8))
        if i > literal(0)
        if literal(12) / i > literal(2)
      do out(i) := literal(12) / i
    }
    val expected = kernel("guardOrder", signature) { _ =>
      gpuFor("i", literal(0), literal(8)) { i =>
        when(i > literal(0)) {
          when(literal(12) / i > literal(2)) { out(i) := literal(12) / i }
        }
      }
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("filtered mapping and folding preserve explicit conditional-update IR exactly"):
    val source = input[Int]("source")
    val out = output[Int]("out")
    val signature = params(source, out)
    val actual = kernel("guardedFold", signature) { _ =>
      val result = gpuRange("i", literal(0), literal(8)).map(i => source(i).read)
        .filter(_ !== literal(0)).map(x => literal(12) / x).filter(_ > literal(1))
        .foldLeft("state", literal(5))((acc, x) => acc * literal(2) + x)
      out(literal(0)) := result
    }
    val expected = kernel("guardedFold", signature) { _ =>
      val state = local("state", literal(5))
      gpuFor("i", literal(0), literal(8)) { i =>
        when(source(i).read !== literal(0)) {
          val x = literal(12) / source(i).read
          when(x > literal(1)) { state := state.read * literal(2) + x }
        }
      }
      out(literal(0)) := state.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("filtered callbacks are lazy and stage once per terminal even for empty bounds"):
    var maps = 0
    var predicates = 0
    var bodies = 0
    val pipeline = gpuRange("i", literal(0), literal(0))
      .map { i => maps += 1; i }.filter { i => predicates += 1; i > literal(0) }
    assertEquals((maps, predicates), (0, 0))
    kernel("stagingCounts") {
      pipeline.foreach { _ => bodies += 1 }
      pipeline.foreach { _ => bodies += 1 }
    }
    assertEquals((maps, predicates, bodies), (2, 2, 2))

  test("predicate map and step callbacks reject captured statement effects"):
    for operation <- Vector("predicate", "map", "step") do
      val error = intercept[DslError] {
        kernel("hiddenFilterEffect") {
          val selected = gpuRange("i", literal(0), literal(8)).filter { i =>
            if operation == "predicate" then barrier()
            i > literal(0)
          }.map { i =>
            if operation == "map" then barrier()
            i
          }
          selected.foldLeft("state", literal(0)) { (acc, x) =>
            if operation == "step" then barrier()
            acc + x
          }
          ()
        }
      }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("filtered loops retain lexical validation and barrier divergence warnings"):
    val out = output[Int]("out")
    var escaped = Option.empty[Expr[Int]]
    val invalid = kernel("escapedFilteredLocal", params(out)) { _ =>
      gpuRange("i", literal(0), literal(8)).filter(_ > literal(0)).foreach { i =>
        escaped = Some(let("temporary", i))
      }
      out(literal(0)) := escaped.get
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val divergent = kernel("filteredBarrier") {
      gpuRange("i", literal(0), literal(8)).filter(_ < threadIdx.x).foreach(_ => barrier())
    }
    assertEquals(KernelValidator.validate(divergent).warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge))

  test("filters preserve their call-site span independently of the terminal loop"):
    val filterSpan = SourceSpan("Filter.scala", 10, 2, 10, 30)
    val loopSpan = SourceSpan("Filter.scala", 12, 2, 12, 30)
    val pipeline = gpuRange("i", literal(0), literal(8))
      .filter(_ > literal(0))(using DslSourcePosition(filterSpan))
    val definition = kernel("filterSpans") {
      pipeline.foreach(_ => ())(using summon[BlockBuilder], DslSourcePosition(loopSpan))
    }
    val loop = definition.body.statements.head.asInstanceOf[ForLoop]
    assertEquals(loop.span, loopSpan)
    assertEquals(loop.body.statements.head.span, filterSpan)

  test("filter predicates require device Booleans and filtered sums are not implied"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(8)).filter(_ => true)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(8)).filter(_ > literal(0)).sum(literal(0))
    """).nonEmpty)

  test("filtered pipelines support mapping and ordered scalar folds"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("filteredFold") {
        gpuRange("i", literal(0), literal(8))
          .map(_ - literal(4)).filter(_ > literal(0)).map(_ * literal(2))
          .foldLeft("total", literal(0))(_ + _)
        ()
      }
    """), Nil)
