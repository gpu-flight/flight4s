package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class FlatMapSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("multiple staged generators and guards support Scala for yield"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("nestedFor") {
        val values = for
          i <- gpuRange("i", literal(0), literal(5))
          if i > literal(0)
          j <- gpuRange("j", literal(0), i)
          if j < literal(2)
        yield i * literal(5) + j
        values.foreach(_ => ())
        values.foldLeft("total", literal(0))(_ + _)
        ()
      }
    """), Nil)

  test("dependent generators and guards lower exactly to explicitly nested CUDA loops"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("nestedGuards", signature) { _ =>
      val elements = for
        i <- gpuRange("i", literal(0), literal(5))
        if i > literal(0)
        j <- gpuRange("j", literal(0), i)
        if j < literal(2)
      yield i * literal(5) + j
      elements.foreach(index => out(index) := index)
    }
    val expected = kernel("nestedGuards", signature) { _ =>
      gpuFor("i", literal(0), literal(5)) { i =>
        when(i > literal(0)) {
          gpuFor("j", literal(0), i) { j =>
            when(j < literal(2)) {
              val index = i * literal(5) + j
              out(index) := index
            }
          }
        }
      }
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("flatMap results support later expansion and preserve serial scalar fold placement"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("tripleFold", signature) { _ =>
      val result = gpuRange("i", literal(0), literal(4)).map(_ + literal(1))
        .flatMap(i => gpuRange("j", literal(0), i).map(j => i + j))
        .flatMap(x => gpuRange("k", literal(0), literal(2)).map(k => x + k))
        .map(_ * literal(2)).filter(_ > literal(3))
        .foldLeft("state", literal(7))((acc, x) => acc + x)
      out(literal(0)) := result
    }
    val expected = kernel("tripleFold", signature) { _ =>
      val state = local("state", literal(7))
      gpuFor("i", literal(0), literal(4)) { i =>
        val shifted = i + literal(1)
        gpuFor("j", literal(0), shifted) { j =>
          gpuFor("k", literal(0), literal(2)) { k =>
            val x = (shifted + j + k) * literal(2)
            when(x > literal(3)) { state := state.read + x }
          }
        }
      }
      out(literal(0)) := state.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("pair fold initializes once outside both loops and snapshots inside the inner loop"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("nestedPair", signature) { _ =>
      val (a, b) = gpuRange("i", literal(0), literal(4))
        .flatMap(i => gpuRange("j", literal(0), i))
        .foldLeft("state", (literal(0), literal(1)))((s, x) => (s._2, s._1 + x))
      out(literal(0)) := a
      out(literal(1)) := b
    }
    val expected = kernel("nestedPair", signature) { _ =>
      val a = local("state_0", literal(0))
      val b = local("state_1", literal(1))
      gpuFor("i", literal(0), literal(4)) { i =>
        gpuFor("j", literal(0), i) { j =>
          val nextA = let("state_next_0", b.read)
          val nextB = let("state_next_1", a.read + j)
          a := nextA
          b := nextB
        }
      }
      out(literal(0)) := a.read
      out(literal(1)) := b.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("inner factories and later callbacks stage once per terminal even for empty bounds"):
    var expansions = 0
    var maps = 0
    var predicates = 0
    var bodies = 0
    val values = gpuRange("i", literal(0), literal(0)).flatMap { i =>
      expansions += 1
      gpuRange("j", literal(0), i)
    }.map { x => maps += 1; x }.filter { x => predicates += 1; x > literal(0) }
    assertEquals((expansions, maps, predicates), (0, 0, 0))
    kernel("nestedStaging") {
      values.foreach(_ => bodies += 1)
      values.foreach(_ => bodies += 1)
    }
    assertEquals((expansions, maps, predicates, bodies), (2, 2, 2, 2))

  test("inner factories maps predicates and fold steps cannot emit captured builder effects"):
    for operation <- Vector("factory", "map", "predicate", "step") do
      val error = intercept[DslError] {
        kernel("badFlatMapEffect") {
          gpuRange("i", literal(0), literal(4)).flatMap { i =>
            if operation == "factory" then barrier()
            gpuRange("j", literal(0), i)
          }.map { x =>
            if operation == "map" then barrier()
            x
          }.filter { x =>
            if operation == "predicate" then barrier()
            x > literal(0)
          }.foldLeft("state", literal(0)) { (acc, x) =>
            if operation == "step" then barrier()
            acc + x
          }
          ()
        }
      }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("nested traversal retains lexical name and varying-bound barrier checks"):
    val conflict = kernel("nestedIndexConflict") {
      gpuRange("i", literal(0), literal(4)).flatMap(i => gpuRange("i", literal(0), i)).foreach(_ => ())
    }
    assertEquals(KernelValidator.validate(conflict).errors.map(_.code),
      Vector(ValidationCode.LoopIndexConflictsWithBinding))
    val varying = kernel("nestedBarrier") {
      gpuRange("i", literal(0), literal(4)).flatMap(_ => gpuRange("j", literal(0), threadIdx.x))
        .foreach(_ => barrier())
    }
    assertEquals(KernelValidator.validate(varying).warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge))
    var escaped = Option.empty[Expr[Int]]
    val out = output[Int]("out")
    val invalid = kernel("escapedNestedIndex", params(out)) { _ =>
      gpuRange("i", literal(0), literal(4)).flatMap(i => gpuRange("j", literal(0), i))
        .foreach(j => escaped = Some(j))
      out(literal(0)) := escaped.get
    }
    assert(!KernelValidator.validate(invalid).isValid)

  test("loop and filter spans preserve their own staging locations"):
    val loopSpan = SourceSpan("Nested.scala", 8, 2, 8, 70)
    val filterSpan = SourceSpan("Nested.scala", 5, 2, 5, 70)
    val values = gpuRange("i", literal(0), literal(4)).flatMap(i => gpuRange("j", literal(0), i))
      .filter(_ > literal(0))(using DslSourcePosition(filterSpan))
    val definition = kernel("nestedSpans") {
      values.foreach(_ => ())(using summon[BlockBuilder], DslSourcePosition(loopSpan))
    }
    val outer = definition.body.statements.head.asInstanceOf[ForLoop]
    val inner = outer.body.statements.head.asInstanceOf[ForLoop]
    assertEquals(outer.span, loopSpan)
    assertEquals(inner.span, loopSpan)
    assertEquals(inner.body.statements.head.span, filterSpan)

  test("flatMap requires staged traversals and does not imply tuple elements or expression sums"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(4)).flatMap(i => List(i))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val elements = for i <- gpuRange("i", literal(0), literal(4))
                         j <- gpuRange("j", literal(0), i) yield (i, j)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(4)).flatMap(i => gpuRange("j", literal(0), i)).sum(literal(0))
    """).nonEmpty)

  test("mapped flatMap results compose with maps filters and pair folds"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("nestedComposition") {
        gpuRange("i", literal(0), literal(5)).map(_ + literal(1))
          .flatMap(i => gpuRange("j", literal(0), i).map(j => i + j))
          .map(_ * literal(2)).filter(_ > literal(1))
          .foldLeft("state", (literal(0), literal(0))) { (s, x) =>
            (s._1 + literal(1), s._2 + x)
          }
        ()
      }
    """), Nil)
