package flight4s.frontend

import munit.FunSuite
import flight4s.core.ir.*

class ScalaKernelTupleFoldCompilerSuite extends FunSuite:
  private def factory(body: String, setup: String = ""): String = s"""
    package quotedtuplefoldfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    class Definitions:
      $setup
      @experimental
      def definition = kernel("tupleFold", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { p =>
        $body
      }
  """

  private def accepted(body: String): Unit = CompilerHarness.withDirectory { directory =>
    val result = CompilerHarness.compile(directory, "TupleFold", factory(body))
    assertEquals(result.errors, Vector.empty)
    CompilerHarness.withClasses(Seq(result.classes)) { loader =>
      val definition = loader.loadClass("quotedtuplefoldfixture.Definitions")
      val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
      assert(KernelValidator.validate(staged).isValid)
    }
  }

  test("tuple folds accept dependent next fields and immutable result aliases"):
    accepted("""
      val result = deviceRange(0, p._3).foldLeft((1, 2))((state, i) => (state._2, state._1 + i))
      val saved = result
      p._2(0) = saved._1 + result(1)
    """)

  private def rejected(body: String, message: String, setup: String = ""): Unit = CompilerHarness.withDirectory { directory =>
    val result = CompilerHarness.compile(directory, "RejectedTupleFold", factory(body, setup))
    assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
    assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
  }

  test("mixed primitive fields tuple elements identity steps and earlier result seeds compile"):
    accepted("""
      val pairs = deviceRange(0, p._3).map(i => (i, 2.0f, 3.0, i > 0))
      val first = pairs.foldLeft((7, 1.0f, 2.0, false))((state, pair) =>
        (state._1 + pair._1, state._2 - pair._2, state._3 / 2.0 - pair._3, state._4 || pair._4))
      val alias = first
      val second = pairs.foldLeft(alias)((state, pair) => state)
      val scalar = pairs.foldLeft(second._1)((sum, pair) => sum + pair._1)
      val mapped = pairs.map(pair => second)
      mapped.foreach(state => p._2(0) = if state._4 && state._2 > 0.0f && state._3 > 0.0 then scalar else state._1)
    """)

  test("Tuple1 and Tuple22 states retain all fields"):
    accepted("val result = deviceRange(0, p._3).foldLeft(Tuple1(7))((state, i) => Tuple1(state._1 - i)); p._2(0) = result(0)")
    val seed = (1 to 22).mkString(", ")
    val step = (1 to 22).map(n => s"state._${n % 22 + 1}").mkString(", ")
    accepted(s"val result = deviceRange(0, p._3).foldLeft(($seed))((state, i) => ($step)); p._2(0) = result._1 + result._22")

  test("tuple folds compose nested generators guards plan aliases and nested lexical scopes"):
    accepted("""
      val values = for i <- deviceRange(0, p._3); j <- deviceRange(0, i) if j % 2 == 0 yield (i, j)
      val alias = values
      val result = alias.foldLeft((0, 1))((state, pair) => (state._2 + pair._1, state._1 - pair._2))
      val previous = result
      for i <- 0 until p._3 do
        val result = deviceRange(0, i).foldLeft((i, previous._1))((state, item) => (state._2, state._1 + item))
        p._2(i) = result._1
    """)

  test("projection only callback bindings and tuple traversal seeds compile"):
    accepted("""
      val pairs = deviceRange(0, p._3).map(i => (i, p._1(i)))
      pairs.foreach { pair =>
        val result = deviceRange(0, pair._1).foldLeft(pair)((state, i) => {
          val second = state._2
          val first = state._1
          (second, first + i)
        })
        p._2(pair._1) = result._1 + result._2
      }
    """)

  test("unused and empty tuple folds still validate every seed and step field"):
    rejected("val unused = deviceRange(0, 0).foldLeft((0, host))((state, i) => state)", "captures", "val host = 7")
    rejected("val unused = deviceRange(0, 0).foldLeft((0, 0))((state, i) => (state._1, host))", "captures", "val host = 7")
    rejected("val unused = deviceRange(0, 0).foldLeft((0, 0))((state, i) => (state._1, helper(i)))", "captures", "def helper(i: Int): Int = i")

  test("empty nested nonprimitive and larger state tuples are rejected"):
    for seed <- Vector("EmptyTuple", "(0, (1, 2))", "(0, \"host\")", "Tuple1(p._1)") do
      rejected(s"val result = deviceRange(0, p._3).foldLeft($seed)((state, i) => state)", "flat nonempty primitive tuples")
    val fields = Vector.fill(23)("0").mkString(", ")
    rejected(s"val result = deviceRange(0, p._3).foldLeft(($fields))((state, i) => state)", "at most 22 fields")

  test("host tuple seeds helpers lookalike constructors and conversions are rejected"):
    rejected("val result = deviceRange(0, p._3).foldLeft(host)((state, i) => state)", "direct standard tuple constructor", "val host = (0, 1)")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => helper(i))", "direct standard tuple constructor", "def helper(i: Int): (Int, Int) = (i, i)")
    rejected("val result = deviceRange(0, p._3).foldLeft(Tuple2(0, 1))((state, i) => state)", "direct standard tuple constructor", "object Tuple2 { def apply(a: Int, b: Int): (Int, Int) = (a, b) }")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1.0f))((state, i) => (i, i.toFloat))", "primitive operation")

  test("tuple seeds and steps reject effects and arbitrary expression blocks"):
    rejected("val result = deviceRange(0, p._3).foldLeft((0, { p._2(0) = 1; 1 }))((state, i) => state)", "expression blocks")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => (state._2, { p._2(0) = i; state._1 }))", "expression blocks")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => { val next = state._1 + i; (next, state._2) })", "direct standard tuple constructor")

  test("tuple results must be immutable and cannot be reassigned or constructed as general locals"):
    rejected("var result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => state)", "must be immutable")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => state); var alias = result", "primitive locals")
    rejected("val pair = (0, 1); val result = deviceRange(0, p._3).foldLeft(pair)((state, i) => state)", "primitive locals")

  test("conditional tuples stored callbacks and embedded tuple terminals are rejected"):
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => if i > 0 then state else (1, 0))", "direct standard tuple constructor")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))(callback)", "literal matching two-parameter", "val callback: ((Int, Int), Int) => (Int, Int) = (state, i) => state")
    rejected("p._2(0) = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => state)._1", "ScalaKernel")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => deviceRange(0, i).foldLeft(state)((next, j) => next))", "direct standard tuple constructor")

  test("field types and literal bounds remain checked and dynamic projections remain rejected"):
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1.0f))((state, i) => state); p._2(0) = result._2", "Required")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => state); p._2(0) = result(2)", "Required")
    rejected("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => state); p._2(0) = result(p._3).asInstanceOf[Int]", "ScalaKernel")

  test("tuple fold kernels preserve exact launch types across separately compiled callers"):
    CompilerHarness.withDirectory { directory =>
      val definition = CompilerHarness.compile(directory, "TupleFoldDefinition", factory("val result = deviceRange(0, p._3).foldLeft((0, 1))((state, i) => (state._2, state._1 + i)); p._2(0) = result._1"))
      assertEquals(definition.errors, Vector.empty)
      val source = """
        package quotedtuplefoldfixture
        import scala.annotation.experimental
        import flight4s.core.ir.{DeviceBuffer, Kernel}
        class Caller:
          @experimental
          def definition: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int)] = new Definitions().definition
      """
      assertEquals(CompilerHarness.compile(directory, "TupleFoldCaller", source, dependencies = Seq(definition.classes)).errors, Vector.empty)
      val wrong = CompilerHarness.compile(directory, "WrongTupleFoldCaller", source.replace("DeviceBuffer[Int]", "DeviceBuffer[Float]"), dependencies = Seq(definition.classes))
      assert(wrong.errors.exists(_.contains("Required")), wrong.errors.mkString("\n"))
    }
