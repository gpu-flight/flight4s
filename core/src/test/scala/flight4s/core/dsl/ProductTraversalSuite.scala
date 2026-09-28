package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class ProductTraversalSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
  private case class Entry(index: Expr[Int], next: Expr[Int]) derives ProductFoldState

  test("named expression products compose through maps guards and nested generators"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.dsl.ProductFoldState
      import flight4s.core.ir.Expr
      case class Entry(index: Expr[Int], weight: Expr[Float]) derives ProductFoldState
      kernel("namedElements") {
        val entries = for
          entry <- gpuRange("i", literal(0), literal(8))
            .map(i => Entry(i, convert.i32ToF32(i)))
          if entry.index > literal(0)
          inner <- gpuRange("j", literal(0), entry.index)
            .map(j => Entry(j, entry.weight))
        yield Entry(inner.index, inner.weight + literal(1.0f))
        val total: Expr[Float] = entries.foldLeft("total", literal(0.0f))((sum, entry) => sum + entry.weight)
        ()
      }
    """), Nil)

  test("named values preserve guards and generate exactly the equivalent scalar IR and CUDA"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("productGuard", signature) { _ =>
      gpuRange("i", literal(0), literal(9)).by(2)
        .map(i => Entry(i, i + literal(1)))
        .filter(_.index > literal(0))
        .map(e => Entry(e.index, literal(12) / e.index + e.next))
        .foreach(e => out(e.index) := e.next)
    }
    val expected = kernel("productGuard", signature) { _ =>
      gpuFor("i", literal(0), literal(9), 2) { i =>
        when(i > literal(0)) { out(i) := literal(12) / i + (i + literal(1)) }
      }
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("scalar tuple and product traversals can enter and leave named products"):
    val definition = kernel("productTransitions") {
      val range = gpuRange("i", literal(0), literal(8))
      val mapped = range.map(_ + literal(1))
      val filtered = range.filter(_ > literal(0))
      val nested = range.flatMap(i => gpuRange("j", literal(0), i))
      Vector(range, mapped, filtered, nested).zipWithIndex.foreach { (traversal, n) =>
        traversal.map(i => Entry(i, i + literal(1))).map(_.next)
          .foldLeft(s"sum$n", literal(0))(_ + _)
      }
      range.map(i => (i, i)).map(p => Entry(p._1, p._2))
        .map(e => (e.index, e.next)).map(p => p._1 + p._2)
        .foldLeft("tupleMapped", literal(0))(_ + _)
      range.flatMap(i => gpuRange("j", literal(0), i).map(j => Entry(i, j)))
        .flatMap(e => gpuRange("k", literal(0), e.next).map(k => (e.index, k)))
        .flatMap(p => gpuRange("l", literal(0), p._2).map(l => Entry(p._1, l)))
        .flatMap(e => gpuRange("m", literal(0), e.next))
        .foldLeft("mixedNested", literal(0))(_ + _)
      ()
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    assert(CudaCodegen.generate(definition).isRight)

  test("product elements reach scalar tuple and simultaneous named fold states"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("productFolds", signature) { _ =>
      val entries = gpuRange("i", literal(0), literal(8)).map(i => Entry(i, i + literal(1)))
      val scalar = entries.foldLeft("scalar", literal(0))((s, e) => s + e.index + e.next)
      val pair = entries.foldLeft("pair", (literal(0), literal(1)))((s, e) => (s._2, s._1 + e.next))
      val triple = entries.foldLeft("triple", (literal(0), literal(1), literal(2)))((s, e) =>
        (s._2, s._3, s._1 + e.index))
      val named = entries.foldLeft("named", Entry(literal(0), literal(1)))((s, e) =>
        Entry(s.next, s.index + e.next))
      out(literal(0)) := scalar + pair._1 + triple._2 + named.next
    }
    val expected = kernel("productFolds", signature) { _ =>
      val entries = gpuRange("i", literal(0), literal(8)).map(i => (i, i + literal(1)))
      val scalar = entries.foldLeft("scalar", literal(0))((s, e) => s + e._1 + e._2)
      val pair = entries.foldLeft("pair", (literal(0), literal(1)))((s, e) => (s._2, s._1 + e._2))
      val triple = entries.foldLeft("triple", (literal(0), literal(1), literal(2)))((s, e) =>
        (s._2, s._3, s._1 + e._1))
      val named = entries.foldLeft("named", Entry(literal(0), literal(1)))((s, e) =>
        Entry(s.next, s.index + e._2))
      out(literal(0)) := scalar + pair._1 + triple._2 + named.next
    }
    assertEquals(KernelValidator.validate(actual).errors, Vector.empty)
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("callbacks and constructors stage once per terminal even for an empty device range"):
    var constructors = 0
    var maps = 0
    var predicates = 0
    var steps = 0
    case class Counted(index: Expr[Int]) derives ProductFoldState:
      constructors += 1
    kernel("emptyProducts") {
      val entries = gpuRange("i", literal(0), literal(0)).map { i => maps += 1; Counted(i) }
        .filter { e => predicates += 1; e.index > literal(0) }
      for name <- Vector("first", "second") do
        entries.foldLeft(name, literal(0)) { (sum, e) => steps += 1; sum + e.index }
    }
    assertEquals((constructors, maps, predicates, steps), (2, 2, 2, 2))

  test("all product callbacks reject captured statement emission and recover the staging guard"):
    for operation <- Vector("map", "productMap", "tupleMap", "scalarMap", "filter", "productFlatMap",
        "tupleFlatMap", "scalarFlatMap", "fold") do
      val error = intercept[DslError] {
        kernel("badProductCallbacks") {
          gpuRange("i", literal(0), literal(4))
            .map { i => if operation == "map" then barrier(); Entry(i, i) }
            .map { e => if operation == "productMap" then barrier(); e }
            .filter { e => if operation == "filter" then barrier(); e.index > literal(0) }
            .flatMap { e =>
              if operation == "productFlatMap" then barrier()
              gpuRange("j", literal(0), e.index).map(j => Entry(e.index, j))
            }
            .map { e => if operation == "tupleMap" then barrier(); (e.index, e.next) }
            .map(p => Entry(p._1, p._2))
            .flatMap { e =>
              if operation == "tupleFlatMap" then barrier()
              gpuRange("k", literal(0), e.next).map(k => (e.index, k))
            }
            .map(p => Entry(p._1, p._2))
            .map { e => if operation == "scalarMap" then barrier(); e.index }
            .map(i => Entry(i, i))
            .flatMap { e =>
              if operation == "scalarFlatMap" then barrier()
              gpuRange("l", literal(0), e.next)
            }
            .map(i => Entry(i, i))
            .foldLeft("sum", literal(0)) { (s, e) =>
              if operation == "fold" then barrier()
              s + e.index
            }
          ()
        }
      }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)
    val error = intercept[DslError] {
      kernel("badProductConstructor") {
        case class Bad(index: Expr[Int]) derives ProductFoldState:
          barrier()
        gpuRange("i", literal(0), literal(4)).map(i => Bad(i)).foreach(_ => ())
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)
    assert(KernelValidator.validate(kernel("afterProducts") { barrier() }).isValid)

  test("host fields empty products and nested products cannot acquire the required evidence"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.ProductFoldState
      import flight4s.core.ir.Expr
      case class Bad(index: Expr[Int], host: Int) derives ProductFoldState
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.ProductFoldState
      case class Empty() derives ProductFoldState
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.ProductFoldState
      import flight4s.core.ir.Expr
      case class Inner(index: Expr[Int]) derives ProductFoldState
      case class Outer(inner: Inner) derives ProductFoldState
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      case class MissingEvidence(index: Expr[Int])
      gpuRange("i", literal(0), literal(4)).map(i => MissingEvidence(i))
    """).nonEmpty)

  test("a named product does not silently snapshot its expression fields"):
    val out = output[Int]("out")
    val signature = params(out)
    val definition = kernel("liveProductFields", signature) { _ =>
      val current = local("current", literal(1))
      gpuRange("i", literal(0), literal(1)).map(i => Entry(i, current.read)).foreach { e =>
        current := literal(7)
        out(e.index) := e.next
      }
    }
    val expected = kernel("liveProductFields", signature) { _ =>
      val current = local("current", literal(1))
      gpuFor("i", literal(0), literal(1)) { i =>
        current := literal(7)
        out(i) := current.read
      }
    }
    assertEquals(definition.ir, expected.ir)
    assertEquals(CudaCodegen.generate(definition), CudaCodegen.generate(expected))
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
