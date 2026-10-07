package flight4s.frontend

import munit.FunSuite
import flight4s.core.ir.*

class ScalaKernelNamedTupleCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = ""): String = s"""
    package quotednamedtuplefixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("namedTupleFold", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = ""): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "NamedTuple", factory(body, schema))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotednamedtuplefixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
      }
    }

  test("named tuple folds preserve dependent fields and immutable aliases"):
    accepted("""
      val result = deviceRange(0, p._3).foldLeft((sum = 1, count = 2))((state, i) => (sum = state.count, count = state.sum + i))
      val saved = result
      p._2(0) = saved.sum + result.count
    """)

  private def rejected(body: String, message: String, schema: String = ""): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedNamedTuple", factory(body, schema))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("mixed named fields compose maps guards folds identity and reused seeds"):
    accepted("""
      val values = deviceRange(0, p._3).map(i => (index = i, amount = 2.0f, scale = 3.0, active = i > 0))
      val first = values.withFilter(item => item.active).foldLeft((sum = 1.0f, count = 0, value = 2.0, found = false))((state, item) =>
        (sum = state.sum - item.amount, count = state.count + item.index, value = state.value / 2.0 - item.scale, found = state.found || item.active))
      val second = values.foldLeft(first)((state, item) => state)
      val result = values.map(item => second).foldLeft(0)((total, state) => total + state.count)
      p._2(0) = if second.found && second.sum > 0.0f && second.value > 0.0 then result else first.count
    """)

  test("named elements preserve nested generators guards and projection-only callback blocks"):
    accepted("""
      val values = for i <- deviceRange(0, p._3); j <- deviceRange(0, i) if j % 2 == 0 yield (outer = i, inner = j)
      val alias = values
      val result = alias.foldLeft((left = 0, right = 1))((state, item) => {
        val right = state.right
        val index = item.inner
        (left = right + item.outer, right = state.left - index)
      })
      values.foreach { item =>
        val saved = item
        val result = deviceRange(0, item.inner).foldLeft(saved)((state, i) => (outer = state.inner, inner = state.outer + i))
        p._2(item.outer) = result.inner
      }
    """)

  test("named tuples interoperate with ordinary tuple elements and states"):
    accepted("""
      val pairs = deviceRange(0, p._3).map(i => (i, p._1(i)))
      val named = pairs.map(pair => (index = pair._1, item = pair._2))
      val ordinary = named.map(pair => (pair.index, pair.item))
      val result = ordinary.foldLeft((sum = 0, count = 0))((state, pair) => (sum = state.sum + pair._2, count = state.count + 1))
      val back = named.foldLeft((result.sum, result.count))((state, pair) => (state._2, state._1 + pair.item))
      p._2(0) = back._1
    """)

  test("named tuple aliases and one through twenty-two fields compile"):
    accepted("val result: Stats = deviceRange(0, p._3).foldLeft[Stats]((sum = 0, count = 1))((state, i) => (sum = state.count, count = state.sum + i)); p._2(0) = result.sum",
      "type Stats = (sum: Int, count: Int)")
    accepted("val result = deviceRange(0, p._3).foldLeft((sum = 7))((state, i) => (sum = state.sum - i)); p._2(0) = result.sum")
    val seed = (1 to 22).map(i => s"field$i = $i").mkString(", ")
    val step = (1 to 22).map(i => s"field$i = state.field${i % 22 + 1}").mkString(", ")
    accepted(s"val result = deviceRange(0, p._3).foldLeft(($seed))((state, i) => ($step)); p._2(0) = result.field1 + result.field22")

  test("unused and empty plans validate all named fields"):
    rejected("val unused = deviceRange(0, 0).map(i => (first = i, second = Host.value))", "captures", "object Host { val value = 7 }")
    rejected("val unused = deviceRange(0, 0).foldLeft((sum = 0, count = Host.value))((state, i) => state)", "captures", "object Host { val value = 7 }")
    rejected("val unused = deviceRange(0, 0).foldLeft((sum = 0, count = 0))((state, i) => (sum = state.sum, count = Host.value))", "captures", "object Host { val value = 7 }")

  test("named tuples reject nested nonprimitive and oversized fields"):
    for seed <- Vector("(sum = 0, nested = (1, 2))", "(sum = 0, text = \"host\")", "(data = p._1)") do
      rejected(s"val result = deviceRange(0, p._3).foldLeft($seed)((state, i) => state)", "flat nonempty primitive tuples")
    val seed = (1 to 23).map(i => s"field$i = 0").mkString(", ")
    rejected(s"val result = deviceRange(0, p._3).foldLeft(($seed))((state, i) => state)", "at most 22 fields")

  test("host seeds factories effects and conditional whole tuples are rejected"):
    rejected("val result = deviceRange(0, p._3).foldLeft(Host.seed)((state, i) => state)", "direct standard tuple constructor", "object Host { val seed = (sum = 0, count = 0) }")
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => Host.next(i))", "direct standard tuple constructor", "object Host { def next(i: Int) = (sum = i, count = i) }")
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => (sum = { p._2(0) = i; state.sum }, count = i))", "expression blocks")
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => if i > 0 then state else (sum = 1, count = 2))", "direct standard tuple constructor")

  test("general named locals mutable states and embedded folds remain rejected"):
    rejected("val pair = (sum = 0, count = 0)", "primitive locals")
    rejected("var result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => state)", "must be immutable")
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => state); var alias = result", "primitive locals")
    rejected("p._2(0) = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => state).sum", "ScalaKernel")

  test("Scala checks field names order and types before lowering"):
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => (count = state.count, sum = state.sum)); p._2(0) = result.sum", "Required")
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => state); p._2(0) = result.missing", "missing")
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0.0f, count = 0))((state, i) => state); p._2(0) = result.sum", "Required")

  test("literal named tuple indices work but dynamic indexing and tuple transformations do not"):
    accepted("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => (sum = state(1), count = state(0) + i)); p._2(0) = result(0)")
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => state); p._2(0) = result(p._3).asInstanceOf[Int]", "ScalaKernel")
    rejected("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => state); val tuple = result.toTuple", "primitive locals")

  test("malformed manually encoded named tuple types are rejected"):
    for names <- Vector("(\"sum\", \"sum\")", "Tuple1[\"sum\"]", "(Int, Int)", "(\"sum\", \"count\", Int)") do
      rejected("val result = deviceRange(0, p._3).foldLeft[Stats]((0, 1))((state, i) => state)",
        "named tuple labels", s"type Stats = scala.NamedTuple.NamedTuple[$names, (Int, Int)]")

  test("ordinary case classes remain rejected rather than losing constructor effects"):
    for schema <- Vector("final case class State(sum: Int, count: Int)",
        "final case class State(sum: Int, count: Int) { require(sum >= 0) }",
        "final case class State(sum: Int, count: Int); object State { def apply(sum: Int, count: Int): State = { println(sum); new State(sum, count) } }") do
      rejected("val result = deviceRange(0, p._3).foldLeft(State(0, 0))((state, i) => state)", "primitive locals", schema)

  test("named tuple kernels retain exact launch types across separate callers"):
    CompilerHarness.withDirectory { directory =>
      val definition = CompilerHarness.compile(directory, "NamedDefinition", factory("val result = deviceRange(0, p._3).foldLeft((sum = 0, count = 0))((state, i) => (sum = state.sum + i, count = state.count + 1)); p._2(0) = result.sum"))
      assertEquals(definition.errors, Vector.empty)
      val source = """
        package quotednamedtuplefixture
        import scala.annotation.experimental
        import flight4s.core.ir.{DeviceBuffer, Kernel}
        class Caller:
          @experimental
          def definition: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int)] = new Definitions().definition
      """
      assertEquals(CompilerHarness.compile(directory, "NamedCaller", source, dependencies = Seq(definition.classes)).errors, Vector.empty)
      val wrong = CompilerHarness.compile(directory, "WrongNamedCaller", source.replace("DeviceBuffer[Int]", "DeviceBuffer[Float]"), dependencies = Seq(definition.classes))
      assert(wrong.errors.exists(_.contains("Required")), wrong.errors.mkString("\n"))
    }
