package flight4s.frontend

import munit.FunSuite
import flight4s.core.ir.*

class ScalaKernelTupleCompilerSuite extends FunSuite:
  private def factory(body: String, setup: String = ""): String = s"""
    package quotedtuplefixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    class Definitions:
      $setup
      @experimental
      def definition = kernel("tupleTraversal", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { p =>
        $body
      }
  """

  private def accepted(body: String): Unit = CompilerHarness.withDirectory { directory =>
    val result = CompilerHarness.compile(directory, "TupleTraversal", factory(body))
    assertEquals(result.errors, Vector.empty)
    CompilerHarness.withClasses(Seq(result.classes)) { loader =>
      val definition = loader.loadClass("quotedtuplefixture.Definitions")
      val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
      assert(KernelValidator.validate(staged).isValid)
    }
  }

  test("flat tuple maps compose guards scalar maps foreach and scalar folds"):
    accepted("""
      val pairs = for i <- deviceRange(0, p._3) yield (i, p._1(i))
      val alias = pairs
      alias.withFilter(pair => pair._1 % 2 == 0).foreach { pair => p._2(pair._1) = pair._2 }
      val total = pairs.map(pair => pair._2 + pair._1).foldLeft(7)((sum, item) => sum - item)
      val other = pairs.foldLeft(total)((sum, pair) => sum - pair._2)
      p._2(0) = other
    """)

  private def rejected(body: String, message: String, setup: String = ""): Unit = CompilerHarness.withDirectory { directory =>
    val result = CompilerHarness.compile(directory, "RejectedTupleTraversal", factory(body, setup))
    assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
    assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
  }

  test("tupled callback lambdas literal indices and immutable aliases compile"):
    accepted("""
      val pairs = deviceRange(0, p._3).map(i => (i, p._1(i)))
      pairs.withFilter((index, item) => index >= 0 && item > 0).foreach { (index, item) => p._2(index) = item }
      pairs.map((index, item) => (item, index)).flatMap((item, index) => deviceRange(0, index).map(j => item + j))
        .foreach(item => p._2(0) = item)
      pairs.foreach { pair =>
        val alias = pair
        p._2(alias(0)) = alias(1)
      }
    """)

  test("tuple identity and shape changing maps retain primitive field types"):
    accepted("""
      val values = deviceRange(0, p._3).map(i => (i, 2.0f, 3.0, i > 0))
      val copied = values.map(pair => pair)
      val changed = copied.map(pair => (pair._4, pair._3 / 2.0, pair._2 * 2.0f, pair._1 + 1))
      val total = changed.foldLeft(0)((sum, pair) => if pair._1 && pair._2 > 0.0 && pair._3 > 0.0f then sum + pair._4 else sum)
      p._2(0) = total
    """)

  test("Tuple1 and maximum standard tuple arity compile"):
    accepted("val values = deviceRange(0, p._3).map(i => Tuple1(i)); values.foreach(pair => p._2(pair._1) = pair(0))")
    val fields = (1 to 22).map(n => s"i + $n").mkString(", ")
    accepted(s"val values = deviceRange(0, p._3).map(i => ($fields)); values.foreach(pair => p._2(0) = pair._1 + pair._22)")

  test("flatMap accepts tuple sources tuple results chained generators and scalar terminals"):
    accepted("""
      val fixed = deviceRange(0, 2)
      val pairs = deviceRange(0, p._3).map(i => (i, p._1(i)))
      val nested = pairs.withFilter(pair => pair._1 > 0).flatMap(pair =>
        deviceRange(0, pair._1).map(j => (j, pair._2 + j)))
      val scalar = nested.flatMap(pair => fixed.map(k => pair._2 + k))
      val mixed = for pair <- nested; k <- fixed yield (pair._1, pair._2, k > 0)
      val total = scalar.foldLeft(7)((sum, item) => sum - item)
      mixed.foreach(pair => p._2(pair._1) = if pair._3 then pair._2 else total)
    """)

  test("all primitive scalar accumulator types accept tuple elements"):
    accepted("""
      val pairs = deviceRange(0, p._3).map(i => (i, 2.0f, 3.0, i > 0))
      val f = pairs.foldLeft(7.0f)((sum, pair) => sum - pair._2)
      val d = pairs.foldLeft(7.0)((sum, pair) => sum / 2.0 - pair._3)
      val b = pairs.foldLeft(false)((found, pair) => found || pair._4)
      val i = pairs.foldLeft(7)((sum, pair) => sum - pair._1)
      p._2(0) = if b && f > 0.0f && d > 0.0 then i else 0
    """)

  test("empty nested nonprimitive and oversized tuple elements remain rejected"):
    for result <- Vector("EmptyTuple", "(i, (i, i))", "(i, \"host\")", "Tuple1(p._1)") do
      rejected(s"val unused = deviceRange(0, p._3).map(i => $result)", "pure primitive results")
    val fields = Vector.fill(23)("i").mkString(", ")
    rejected(s"val unused = deviceRange(0, p._3).map(i => ($fields))", "at most 22 fields")

  test("unused tuple fields validate captures helper calls conversions and embedded folds"):
    rejected("val unused = deviceRange(0, p._3).map(i => (i, host))", "captures", "val host = 7")
    rejected("val unused = deviceRange(0, p._3).map(i => (i, helper(i)))", "captures", "def helper(i: Int): Int = i")
    rejected("val unused = deviceRange(0, p._3).map(i => (i, i.toFloat))", "primitive operation")
    rejected("val unused = deviceRange(0, p._3).map(i => (i, deviceRange(0, i).foldLeft(0)(_ + _)))", "directly initialize")
    rejected("val unused = deviceRange(0, p._3).flatMap(i => deviceRange(0, i).map(j => (i, host)))", "captures", "val host = 7")

  test("tuple constructors cannot smuggle side effects or host lookalikes"):
    rejected("val unused = deviceRange(0, p._3).map(i => (i, { p._2(0) = i; i }))", "expression blocks")
    rejected("val unused = deviceRange(0, p._3).map(i => helper(i))", "direct standard tuple constructor", "def helper(i: Int): (Int, Int) = (i, i)")
    rejected("val unused = deviceRange(0, p._3).map(i => Tuple2(i, i))", "direct standard tuple constructor", "object Tuple2 { def apply(a: Int, b: Int): (Int, Int) = (a, b) }")
    rejected("val unused = deviceRange(0, p._3).map(i => if i > 0 then (i, i) else (0, 0))", "direct standard tuple constructor")

  test("tuple projections retain type bounds and reject dynamic indices casts and host products"):
    rejected("val values = deviceRange(0, p._3).map(i => (i, 2.0f)); values.foreach(pair => p._2(0) = pair._2)", "Required")
    rejected("val values = deviceRange(0, p._3).map(i => (i, i)); values.foreach(pair => p._2(0) = pair(2))", "Required")
    rejected("val values = deviceRange(0, p._3).map(i => (i, i)); values.foreach(pair => p._2(0) = pair(p._3).asInstanceOf[Int])", "ScalaKernel")
    rejected("val values = deviceRange(0, p._3).map(i => (i, i)); values.foreach(pair => p._2(0) = pair.productElement(0).asInstanceOf[Int])", "ScalaKernel")
    rejected("val values = deviceRange(0, p._3).map(i => host); values.foreach(pair => p._2(0) = pair._1)", "direct standard tuple constructor", "val host = (1, 2)")

  test("tuple states tuple local construction and mutable tuple aliases remain deferred"):
    rejected("val values = deviceRange(0, p._3).map(i => (i, i)); val result = values.foldLeft((0, 0))((sum, pair) => sum)", "primitive locals")
    rejected("val pair = (1, 2); p._2(0) = pair._1", "primitive locals")
    rejected("val values = deviceRange(0, p._3).map(i => (i, i)); values.foreach { pair => var alias = pair; p._2(0) = alias._1 }", "primitive locals")

  test("separately compiled tuple traversal kernels preserve exact launch argument types"):
    CompilerHarness.withDirectory { directory =>
      val definition = CompilerHarness.compile(directory, "TupleDefinition", factory("val values = deviceRange(0, p._3).map(i => (i, p._1(i))); values.foreach(pair => p._2(pair._1) = pair._2)"))
      assertEquals(definition.errors, Vector.empty)
      val source = """
        package quotedtuplefixture
        import scala.annotation.experimental
        import flight4s.core.ir.{DeviceBuffer, Kernel}
        class Caller:
          @experimental
          def definition: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int)] = new Definitions().definition
      """
      assertEquals(CompilerHarness.compile(directory, "TupleCaller", source, dependencies = Seq(definition.classes)).errors, Vector.empty)
      val wrong = CompilerHarness.compile(directory, "WrongTupleCaller", source.replace("DeviceBuffer[Int]", "DeviceBuffer[Float]"), dependencies = Seq(definition.classes))
      assert(wrong.errors.exists(_.contains("Required")), wrong.errors.mkString("\n"))
    }
