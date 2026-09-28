package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class TupleTraversalSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
  test("map carries typed expression tuples through guards maps and ordered folds"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      kernel("tupleElements") {
        val total: Expr[Float] = gpuRange("i", literal(0), literal(8))
          .map(i => (i, convert.i32ToF32(i)))
          .filter { case (i, x) => i > literal(0) }
          .map { case (i, x) => (i, x + literal(1.0f)) }
          .foldLeft("total", literal(0.0f)) { case (sum, (i, x)) => sum + x }
        ()
      }
    """), Nil)

  test("tuple-valued for comprehensions retain guarded nested generators and value definitions"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      val out = output[Int]("out")
      kernel("tupleFor", params(out)) { _ =>
        val pairs = for
          i <- gpuRange("i", literal(0), literal(8))
          j <- gpuRange("j", literal(0), i)
        yield (i, j)
        for
          (i, j) <- pairs
          if j > literal(0)
          quotient = i / j
          k <- gpuRange("k", literal(0), quotient)
        do out(i) := j + k
      }
    """), Nil)

  test("tuple maps retain existing guards and emit the equivalent scalar program"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("tupleGuard", signature) { _ =>
      gpuRange("i", literal(0), literal(9)).by(2)
        .map(i => (i, i + literal(1)))
        .filter { case (i, _) => i > literal(0) }
        .map { case (i, next) => (i, literal(12) / i, next) }
        .foreach { case (i, quotient, next) => out(i) := quotient + next }
    }
    val expected = kernel("tupleGuard", signature) { _ =>
      gpuFor("i", literal(0), literal(9), 2) { i =>
        when(i > literal(0)) { out(i) := literal(12) / i + (i + literal(1)) }
      }
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("tuple elements reach scalar pair tuple and named-product state"):
    case class State(count: Expr[Int], total: Expr[Int]) derives ProductFoldState
    val out = output[Int]("out")
    val definition = kernel("tupleTerminals", params(out)) { _ =>
      val elements = gpuRange("i", literal(0), literal(8)).map(i => (i, i + literal(1)))
      val scalar = elements.foldLeft("scalar", literal(0)) { case (sum, (i, next)) => sum + i + next }
      val pair = elements.foldLeft("pair", (literal(0), literal(1))) { case (s, (i, next)) =>
        (s._2, s._1 + i + next)
      }
      val triple = elements.foldLeft("triple", (literal(0), literal(1), literal(2))) { case (s, (i, next)) =>
        (s._2, s._3, s._1 + i + next)
      }
      val named = elements.foldLeft("named", State(literal(0), literal(0))) { case (s, (i, next)) =>
        State(s.count + literal(1), s.total + i + next)
      }
      out(literal(0)) := scalar + pair._1 + triple._2 + named.total
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    assert(CudaCodegen.generate(definition).isRight)

  test("tuple mapping is available after every scalar traversal kind"):
    val definition = kernel("tupleFromEach") {
      val range = gpuRange("i", literal(0), literal(8))
      val mapped = range.map(_ + literal(1))
      val filtered = range.filter(_ > literal(0))
      val nested = range.flatMap(i => gpuRange("j", literal(0), i))
      Vector(range, mapped, filtered, nested).zipWithIndex.foreach { (traversal, index) =>
        traversal.map(i => (i, literal(1))).foldLeft(s"sum$index", literal(0)) {
          case (sum, (i, one)) => sum + i + one
        }
      }
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)

  test("tuple callbacks stage once per terminal and preserve expression-effect rejection"):
    var maps = 0
    var predicates = 0
    var steps = 0
    kernel("emptyTupleValues") {
      gpuRange("i", literal(0), literal(0)).map { i => maps += 1; (i, i) }
        .filter { case (i, _) => predicates += 1; i > literal(0) }
        .foldLeft("sum", literal(0)) { case (sum, (i, _)) => steps += 1; sum + i }
      ()
    }
    assertEquals((maps, predicates, steps), (1, 1, 1))
    for operation <- Vector("map", "tupleMap", "filter", "flatMap", "fold") do
      val error = intercept[DslError] {
        kernel("badTupleValues") {
          gpuRange("i", literal(0), literal(4))
            .map { i => if operation == "map" then barrier(); (i, i) }
            .map { pair => if operation == "tupleMap" then barrier(); pair }
            .filter { case (i, _) => if operation == "filter" then barrier(); i > literal(0) }
            .flatMap { case (i, _) =>
              if operation == "flatMap" then barrier()
              gpuRange("j", literal(0), i).map(j => (i, j))
            }
            .foldLeft("sum", literal(0)) { case (sum, (i, j)) =>
              if operation == "fold" then barrier()
              sum + i + j
            }
          ()
        }
      }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)
    assert(KernelValidator.validate(kernel("afterTupleValues") { barrier() }).isValid)

  test("host-valued and empty tuple elements do not type-check"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(4)).map(i => (i, 1))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(4)).map(_ => EmptyTuple)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      gpuRange("i", literal(0), literal(4)).map(i => (i, (i, i)))
    """).nonEmpty)

  test("Tuple1 and tuples above arity 22 remain host-only expression containers"):
    val definition = kernel("largeTupleElements") {
      gpuRange("i", literal(0), literal(4)).map(i => Tuple1(i))
        .map { case Tuple1(i) => (i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i) }
        .map(values => values.head + values.last)
        .foldLeft("sum", literal(0))(_ + _)
      ()
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    assertEquals(definition.body.statements.size, 2)
    assert(CudaCodegen.generate(definition).isRight)
