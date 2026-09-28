package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class TupleFoldSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("foldLeft accepts nonempty tuples with independently typed components"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      kernel("tupleState") {
        val triple: (Expr[Int], Expr[Float], Expr[Boolean]) =
          gpuRange("i", literal(0), literal(8))
            .foldLeft("state", (literal(0), literal(0.0f), literal(false))) { (s, i) =>
              (s._1 + i, s._2 + literal(1.0f), i > literal(2))
            }
        val single: Tuple1[Expr[Int]] = gpuRange("j", literal(0), literal(4))
          .foldLeft("one", Tuple1(literal(0)))((s, j) => Tuple1(s._1 + j))
        ()
      }
    """), Nil)

  test("three components snapshot all next values before any old value is overwritten"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("rotate", signature) { _ =>
      val state = gpuRange("i", literal(0), literal(7)).by(2)
        .foldLeft("state", (literal(1), literal(2), literal(3))) { (s, i) =>
          (s._2, s._3, s._1 + i)
        }
      out(literal(0)) := state._1
      out(literal(1)) := state._2
      out(literal(2)) := state._3
    }
    val expected = kernel("rotate", signature) { _ =>
      val a = local("state_0", literal(1))
      val b = local("state_1", literal(2))
      val c = local("state_2", literal(3))
      gpuFor("i", literal(0), literal(7), 2) { i =>
        val nextA = let("state_next_0", b.read)
        val nextB = let("state_next_1", c.read)
        val nextC = let("state_next_2", a.read + i)
        a := nextA
        b := nextB
        c := nextC
      }
      out(literal(0)) := a.read
      out(literal(1)) := b.read
      out(literal(2)) := c.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("generic pair folds preserve existing IR and CUDA on each traversal kind"):
    val out = output[Int]("out")
    val signature = params(out)
    for kind <- Vector("range", "mapped", "filtered", "flatMapped") do
      def definition(generic: Boolean) = kernel("pairCompatibility", signature) { _ =>
        val range = gpuRange("i", literal(0), literal(8)).by(2)
        val initial = (literal(1), literal(2))
        val step = (s: (Expr[Int], Expr[Int]), i: Expr[Int]) => (s._2, s._1 + i)
        val result = kind match
          case "range" =>
            if generic then range.foldLeft[(Expr[Int], Expr[Int])]("state", initial)(step)
            else range.foldLeft("state", initial)(step)
          case "mapped" =>
            val values = range.map(_ + literal(1))
            if generic then values.foldLeft[(Expr[Int], Expr[Int])]("state", initial)(step)
            else values.foldLeft("state", initial)(step)
          case "filtered" =>
            val values = range.filter(_ > literal(2))
            if generic then values.foldLeft[(Expr[Int], Expr[Int])]("state", initial)(step)
            else values.foldLeft("state", initial)(step)
          case _ =>
            val values = range.flatMap(i => gpuRange("j", literal(0), i))
            if generic then values.foldLeft[(Expr[Int], Expr[Int])]("state", initial)(step)
            else values.foldLeft("state", initial)(step)
        out(literal(0)) := result._1
        out(literal(1)) := result._2
      }
      val before = definition(false)
      val after = definition(true)
      assertEquals(after.ir, before.ir, kind)
      assertEquals(CudaCodegen.generate(after), CudaCodegen.generate(before), kind)

  test("filtered tuple snapshots and stores remain inside their guards"):
    val out = output[Int]("out")
    val signature = params(out)
    val actual = kernel("guardedTriple", signature) { _ =>
      val state = gpuRange("i", literal(0), literal(8)).filter(_ > literal(0))
        .foldLeft("state", (literal(1), literal(2), literal(3))) { (s, i) =>
          (s._2, s._3, s._1 + literal(12) / i)
        }
      out(literal(0)) := state._3
    }
    val expected = kernel("guardedTriple", signature) { _ =>
      val a = local("state_0", literal(1))
      val b = local("state_1", literal(2))
      val c = local("state_2", literal(3))
      gpuFor("i", literal(0), literal(8)) { i =>
        when(i > literal(0)) {
          val nextA = let("state_next_0", b.read)
          val nextB = let("state_next_1", c.read)
          val nextC = let("state_next_2", a.read + literal(12) / i)
          a := nextA
          b := nextB
          c := nextC
        }
      }
      out(literal(0)) := c.read
    }
    assertEquals(actual.ir, expected.ir)
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("callbacks stage once for an empty runtime traversal"):
    var maps = 0
    var predicates = 0
    var steps = 0
    kernel("emptyTupleStages") {
      gpuRange("i", literal(0), literal(0))
        .map { i => maps += 1; i }
        .filter { i => predicates += 1; i > literal(0) }
        .foldLeft("state", (literal(1), literal(2), literal(3))) { (s, _) => steps += 1; s }
      ()
    }
    assertEquals((maps, predicates, steps), (1, 1, 1))

  test("tuple maps predicates and steps reject captured statement effects"):
    for operation <- Vector("map", "filter", "step") do
      val error = intercept[DslError] {
        kernel("badTupleEffect") {
          gpuRange("i", literal(0), literal(4))
            .map { i => if operation == "map" then barrier(); i }
            .filter { i => if operation == "filter" then barrier(); i > literal(0) }
            .foldLeft("state", (literal(1), literal(2), literal(3))) { (s, _) =>
              if operation == "step" then barrier()
              s
            }
          ()
        }
      }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)
    assert(KernelValidator.validate(kernel("afterTupleEffect") { barrier() }).isValid)

  test("tuple state declarations snapshots stores and results retain terminal spans"):
    type State = (Expr[Int], Expr[Float], Expr[Boolean])
    val span = SourceSpan("Tuple.scala", 12, 2, 12, 80)
    var result = Option.empty[State]
    val definition = kernel("tupleSpans") {
      result = Some(gpuRange("i", literal(0), literal(4))
        .foldLeft[State]("state", (literal(1), literal(2.0f), literal(false)))((s, _) => s)(
          using summon[TupleFoldState[State]], summon[BlockBuilder], DslSourcePosition(span)))
    }
    assertEquals(definition.body.statements.size, 4)
    assert(definition.body.statements.forall(_.span == span))
    val loop = definition.body.statements.last.asInstanceOf[ForLoop]
    assertEquals(loop.body.statements.size, 6)
    assert(loop.body.statements.forall(_.span == span))
    assertEquals(result.get.toList.map(_.span), List.fill(3)(span))

  test("tuple results retain lexical lifetime and generated-name validation"):
    var escaped = Option.empty[Expr[Int]]
    val out = output[Int]("out")
    val invalid = kernel("escapedTuple", params(out)) { _ =>
      scoped {
        escaped = Some(gpuRange("i", literal(0), literal(4))
          .foldLeft("state", (literal(1), literal(2), literal(3)))((s, _) => s)._3)
      }
      out(literal(0)) := escaped.get
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val conflict = kernel("tupleConflict") {
      local("state_next_2", literal(0))
      gpuRange("i", literal(0), literal(4))
        .foldLeft("state", (literal(1), literal(2), literal(3)))((s, _) => s)
      ()
    }
    assert(!KernelValidator.validate(conflict).isValid)

  test("empty or host-valued tuples changed arity and changed component types do not compile"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("emptyState") { gpuRange("i", literal(0), literal(4)).foldLeft("s", EmptyTuple)((s, _) => s); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("hostState") { gpuRange("i", literal(0), literal(4)).foldLeft("s", (1, 2, 3))((s, _) => s); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("changedArity") {
        gpuRange("i", literal(0), literal(4)).foldLeft("s", (literal(1), literal(2), literal(3))) {
          (s, _) => (s._1, s._2)
        }; ()
      }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("changedTypes") {
        gpuRange("i", literal(0), literal(4)).foldLeft("s", (literal(1), literal(2.0f), literal(true))) {
          (s, _) => (s._2, s._1, s._3)
        }; ()
      }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("tupleWrite") {
        val state = gpuRange("i", literal(0), literal(4))
          .foldLeft("s", (literal(1), literal(2), literal(3)))((s, _) => s)
        state._3 := literal(4)
      }
    """).nonEmpty)

  test("tuple folds are available through the shared traversal contract"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.dsl.GpuTraversal
      kernel("traversalState") {
        val traversal: GpuTraversal[Int] = gpuRange("i", literal(0), literal(8)).by(2)
          .map(_ + literal(1)).filter(_ > literal(0))
          .flatMap(i => gpuRange("j", literal(0), i))
        val initial = (literal(0), literal(1), literal(2), literal(3))
        traversal.foldLeft("state", initial) { (s, i) => (s._2, s._3, s._4, s._1 + i) }
        ()
      }
    """), Nil)

  test("Scala 3 tuple states are not restricted to 22 components"):
    val definition = kernel("largeTuple") {
      val initial = (
        literal(0), literal(1), literal(2), literal(3), literal(4), literal(5),
        literal(6), literal(7), literal(8), literal(9), literal(10), literal(11),
        literal(12), literal(13), literal(14), literal(15), literal(16), literal(17),
        literal(18), literal(19), literal(20), literal(21), literal(22)
      )
      gpuRange("i", literal(0), literal(4)).foldLeft("state", initial) { (s, i) =>
        s.tail :* (s.head + i)
      }
      ()
    }
    assertEquals(definition.body.statements.size, 24)
    val body = definition.body.statements.last.asInstanceOf[ForLoop].body.statements
    assertEquals(body.size, 46)
    assert(body.take(23).forall(_.isInstanceOf[LocalDeclaration[?]]))
    assert(body.drop(23).forall(_.isInstanceOf[Store[?, ?]]))
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    assert(CudaCodegen.generate(definition).isRight)
