package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class PairFoldSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("ranges accept pair state with independently typed components"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("pairState") {
        gpuRange("i", literal(0), literal(8))
          .foldLeft("state", (literal(0), literal(0.0f))) { (state, i) =>
            (state._1 + i, state._2 + literal(1.0f))
          }
        ()
      }
    """), Nil)

  test("pair steps snapshot both next values before updating either state component"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("pairRecurrence", signature) { _ =>
      val (a, b) = gpuRange("i", literal(0), literal(8))
        .foldLeft("state", (literal(0), literal(1))) { (s, _) => (s._2, s._1 + s._2) }
      out(literal(0)) := a
      out(literal(1)) := b
    }
    val expected = kernel("pairRecurrence", signature) { _ =>
      val a = local("state_0", literal(0))
      val b = local("state_1", literal(1))
      gpuFor("i", literal(0), literal(8)) { _ =>
        val nextA = let("state_next_0", b.read)
        val nextB = let("state_next_1", a.read + b.read)
        a := nextA
        b := nextB
      }
      out(literal(0)) := a.read
      out(literal(1)) := b.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("guarded pair updates and temporary declarations remain inside the GPU guard"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("guardedPair", signature) { _ =>
      val (a, b) = gpuRange("i", literal(0), literal(8)).map(_ - literal(2))
        .filter(_ > literal(0)).foldLeft("state", (literal(1), literal(2))) {
          (s, x) => (s._2, s._1 + literal(12) / x)
        }
      out(literal(0)) := a
      out(literal(1)) := b
    }
    val expected = kernel("guardedPair", signature) { _ =>
      val a = local("state_0", literal(1))
      val b = local("state_1", literal(2))
      gpuFor("i", literal(0), literal(8)) { i =>
        val x = i - literal(2)
        when(x > literal(0)) {
          val nextA = let("state_next_0", b.read)
          val nextB = let("state_next_1", a.read + literal(12) / x)
          a := nextA
          b := nextB
        }
      }
      out(literal(0)) := a.read
      out(literal(1)) := b.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("mapping and pair callbacks stage once per terminal independent of runtime iterations"):
    var maps = 0
    var predicates = 0
    var steps = 0
    kernel("pairStages") {
      val values = gpuRange("i", literal(0), literal(0)).map { i => maps += 1; i }
      values.foldLeft("a", (literal(0), literal(1))) { (s, _) => steps += 1; s }
      values.filter { i => predicates += 1; i > literal(0) }
        .foldLeft("b", (literal(0), literal(1))) { (s, _) => steps += 1; s }
      ()
    }
    assertEquals((maps, predicates, steps), (2, 1, 2))

  test("pair mapping and steps reject captured statement effects and restore the guard"):
    for operation <- Vector("map", "step", "filtered-step") do
      val error = intercept[DslError] {
        kernel("badPairEffect") {
          val values = gpuRange("i", literal(0), literal(4)).map { i =>
            if operation == "map" then barrier()
            i
          }
          if operation == "filtered-step" then
            values.filter(_ > literal(0)).foldLeft("state", (literal(0), literal(1))) {
              (s, _) => barrier(); s
            }
          else
            values.foldLeft("state", (literal(0), literal(1))) { (s, _) =>
              if operation == "step" then barrier()
              s
            }
          ()
        }
      }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)
    assert(KernelValidator.validate(kernel("afterPairFailure") { barrier() }).isValid)

  test("pair results preserve lexical lifetime and generated name conflicts are validated"):
    var escaped = Option.empty[Expr[Int]]
    val out = output[Int]("out")
    val invalid = kernel("escapedPair", params(out)) { _ =>
      scoped {
        escaped = Some(gpuRange("i", literal(0), literal(4))
          .foldLeft("state", (literal(0), literal(1)))((s, _) => s)._1)
      }
      out(literal(0)) := escaped.get
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val conflict = kernel("pairNameConflict") {
      local("state_next_0", literal(0))
      gpuRange("i", literal(0), literal(4)).foldLeft("state", (literal(0), literal(1)))((s, _) => s)
      ()
    }
    assert(!KernelValidator.validate(conflict).isValid)

  test("pair declarations loops temporary stores and results retain terminal source spans"):
    val span = SourceSpan("Pair.scala", 12, 2, 12, 80)
    var result = Option.empty[(Expr[Int], Expr[Float])]
    val definition = kernel("pairSpans") {
      result = Some(gpuRange("i", literal(0), literal(4))
        .foldLeft("state", (literal(0), literal(1.0f)))((s, _) => s)(
          using I32, F32, summon[BlockBuilder], DslSourcePosition(span)
        ))
    }
    assert(definition.body.statements.forall(_.span == span))
    val loop = definition.body.statements(2).asInstanceOf[ForLoop]
    assertEquals(loop.body.statements.size, 4)
    assert(loop.body.statements.forall(_.span == span))
    assertEquals(result.get._1.span, span)
    assertEquals(result.get._2.span, span)

  test("pair component types stay fixed and results expose no assignment operation"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("badPairTypes") {
        gpuRange("i", literal(0), literal(4)).foldLeft("s", (literal(0), literal(1.0f))) {
          (s, _) => (s._2, s._1)
        }
        ()
      }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("badPairWrite") {
        val result = gpuRange("i", literal(0), literal(4))
          .foldLeft("s", (literal(0), literal(1)))((s, _) => s)
        result._1 := literal(3)
      }
    """).nonEmpty)

  test("mapped and guarded ranges retain pair fold syntax"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("composedPair") {
        val values = gpuRange("i", literal(0), literal(8)).map(_ * literal(2))
        values.foldLeft("first", (literal(0), literal(1))) { (s, x) => (s._2, s._1 + x) }
        values.filter(_ > literal(0)).foldLeft("second", (literal(0), literal(1))) {
          (s, x) => (s._2, s._1 + x)
        }
        ()
      }
    """), Nil)
