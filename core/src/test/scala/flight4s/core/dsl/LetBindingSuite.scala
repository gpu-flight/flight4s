package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class LetBindingSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("let stages the same snapshot declaration and reads as an explicit local"):
    val data = inOut[Int]("data")
    val out = output[Int]("out")
    val signature = params(data, out)
    val actual = kernel("snapshot", signature) { _ =>
      val before = let("before", data(literal(0)).read)
      data(literal(0)) := literal(99)
      out(literal(0)) := before + before
    }
    val expected = kernel("snapshot", signature) { _ =>
      val before = local("before", data(literal(0)).read)
      data(literal(0)) := literal(99)
      out(literal(0)) := before.read + before.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("let evaluates a reduction into one local instead of repeating it at each use"):
    val out = output[Int]("out")
    val actual = kernel("storedReduction", params(out)) { _ =>
      val total = let("total", reduceSum("i", literal(0), literal(4), literal(0))(identity))
      out(literal(0)) := total
      out(literal(1)) := total
    }
    assertEquals(actual.body.statements.size, 3)
    assert(actual.body.statements.head.isInstanceOf[LocalDeclaration[?]])

  test("let preserves source spans on the declaration local and read expression"):
    val span = SourceSpan("Snapshot.scala", 12, 4, 12, 42)
    var result = Option.empty[Expr[Int]]
    val definition = kernel("snapshotSpan") {
      result = Some(let("snapshot", literal(7))(
        using summon[flight4s.core.types.CudaType[Int]], summon[BlockBuilder], DslSourcePosition(span)
      ))
    }
    val declaration = definition.body.statements.head.asInstanceOf[LocalDeclaration[Int]]
    assertEquals(declaration.span, span)
    assertEquals(declaration.local.span, span)
    assertEquals(result.get.span, span)

  test("let follows local name and lexical lifetime validation"):
    val out = output[Int]("out")
    var escaped = Option.empty[Expr[Int]]
    val invalid = kernel("escapedLet", params(out)) { _ =>
      scoped { escaped = Some(let("temporary", literal(1))) }
      out(literal(0)) := escaped.get
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val conflict = kernel("duplicateLet") {
      let("temporary", literal(1))
      let("temporary", literal(2))
      ()
    }
    assertEquals(KernelValidator.validate(conflict).errors.map(_.code), Vector(ValidationCode.DuplicateLocalName))

  test("let is a statement-producing binding and cannot hide in expression callbacks"):
    val error = intercept[DslError] {
      kernel("hiddenLet") {
        reduceSum("i", literal(0), literal(4), literal(0))(i => let("temporary", i))
        ()
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("let requires a builder and exposes no mutable place through its result type"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      let("outside", literal(1))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("writeLet") {
        val snapshot = let("snapshot", literal(1))
        snapshot := literal(2)
      }
    """).nonEmpty)
