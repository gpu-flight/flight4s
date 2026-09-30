package flight4s.frontend

import munit.FunSuite
import flight4s.core.ir.*

class ScalaKernelCompilerSuite extends FunSuite:
  private def factory(body: String, setup: String = "", parameters: String =
      "input[Int](\"data\"), output[Int](\"target\"), value[Int](\"count\")"): String = s"""
    package quotedfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    class Definitions:
      $setup
      @experimental
      def definition = kernel("definition", params($parameters)) { p =>
        $body
      }
  """

  private def rejected(source: String, message: String): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Rejected", source)
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("ordinary Scala arithmetic and if lower to a typed kernel without executing the body"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "OrdinaryKernel", """
        package quotedfixture
        import scala.annotation.experimental
        import flight4s.frontend.ScalaKernel.*
        class Definitions:
          @experimental
          def definition = kernel("ordinary", params(input[Float]("data"), output[Float]("target"), value[Int]("count"))) { p =>
            val i = blockIdx.x * blockDim.x + threadIdx.x
            if i < p._3 then
              p._2(i) = p._1(i) * 2.0f
          }
        """)
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
          .asInstanceOf[Kernel[(DeviceBuffer[Float], DeviceBuffer[Float], Int)]]
        assert(KernelValidator.validate(staged).isValid)
        assertEquals(staged.body.statements.size, 2)
        assert(staged.body.statements.last.isInstanceOf[IfThen])
        assertEquals(staged.body.statements.last.asInstanceOf[IfThen].elseBlock, None)
      }
    }

  test("ordinary Scala until loops compile without literal wrappers and construct valid loop IR"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RangeLoop", factory("""
        var total = 0
        for i <- 0 until p._3 do
          val before = total
          total = before + p._1(i)
        p._2(0) = total
      """))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
          .asInstanceOf[Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int)]]
        assert(KernelValidator.validate(staged).isValid)
        assertEquals(staged.body.statements.count(_.isInstanceOf[ForLoop]), 1)
      }
    }

  test("named tuple parameters primitive locals aliases assignments and nested branches compile"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Named", """
        package quotedfixture
        import scala.annotation.experimental
        import flight4s.frontend.ScalaKernel.*
        class Definitions:
          @experimental
          def definition = kernel("named", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { (data, target, count) =>
            val i = blockIdx.x * blockDim.x + threadIdx.x
            if i < count then
              val alias = target
              var total = data(i)
              val before = total
              if i % 2 == 0 then
                total += 3
              else
                total = total - 2
              if total > 0 then
                val total = before + 7
                alias(i) = total
              else
                alias(i) = if before < 0 then total else before
      }
      """)
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
          .asInstanceOf[Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int)]]
        assert(KernelValidator.validate(staged).isValid)
      }
    }

  test("range foreach nested generators loop shadows and compound stores compile and validate"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "NestedRanges", factory("""
        var total = 0
        for i <- 0 until p._3 do
          val before = total
          for i <- 0 until i do total += before + i
        for i <- 0 until p._3; j <- 0 until i do
          if j % 2 == 0 then total += j
        (0 until p._3).foreach { i => p._2(i) = total }
      """))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        def loops(block: Block): Vector[ForLoop] = block.statements.flatMap {
          case loop: ForLoop => Vector(loop) ++ loops(loop.body)
          case branch: IfThen => loops(branch.thenBlock) ++ branch.elseBlock.toVector.flatMap(loops)
          case _ => Vector.empty
        }
        val collected = loops(staged.body)
        assertEquals(collected.size, 5)
        assertEquals(collected.map(_.index.name).distinct.size, 5)
      }
    }

  test("range loops reject guards strides inclusive ranges aliases host factories and nonliteral callbacks"):
    rejected(factory("for i <- 0 to p._3 do p._2(i) = i"), "range loops require")
    rejected(factory("for i <- (0 until p._3).by(2) do p._2(i) = i"), "range loops require")
    rejected(factory("for i <- 0 until p._3 if i > 0 do p._2(i) = i"), "range loops")
    rejected(factory("val indices = 0 until p._3; for i <- indices do p._2(i) = i"), "primitive locals")
    rejected(factory("val values = for i <- 0 until p._3 yield i + 1; p._2(0) = 1"), "primitive locals")
    rejected(factory("for i <- Range(0, p._3) do p._2(i) = i"), "range loops require")
    rejected(factory("for i <- 0 until host do p._2(i) = i", "def host: Int = 3"), "captures")
    rejected(factory("for i <- host until p._3 do p._2(i) = i", "val host = 3"), "captures")
    rejected(factory("for i <- 0 until p._3 do println(i)"), "host effects")
    rejected(factory("(0 until p._3).foreach(callback)", "val callback: Int => Unit = _ => ()"), "literal Int loop-body lambda")
    rejected(factory("(0 until p._3).foreach(callback)", "def callback(i: Int): Unit = ()"), "host effects")
    rejected(factory("for i <- intWrapper(0).until(p._3) do p._2(i) = i", """
      class Pretend:
        def until(end: Int): Range = 0 until end
      def intWrapper(start: Int): Pretend = new Pretend
    """), "range loops require")

  test("primitive parameter and local type matrix compiles and validates"):
    val cases = Vector(("Int", "2", "3"), ("Float", "2.0f", "3.0f"),
      ("Double", "2.0", "3.0"), ("Boolean", "false", "true"))
    cases.foreach { (tpe, initial, other) =>
      val operation = if tpe == "Boolean" then "!total || p._3" else "total + p._3"
      CompilerHarness.withDirectory { directory =>
        val result = CompilerHarness.compile(directory, "Matrix", factory(
          s"val i = threadIdx.x; var total = $initial; val before = total; total = $operation; if before == $other then p._2(i) = total else p._2(i) = before",
          parameters = s"input[$tpe](\"data\"), output[$tpe](\"target\"), value[$tpe](\"scalar\")"))
        assertEquals(result.errors, Vector.empty, tpe)
        CompilerHarness.withClasses(Seq(result.classes)) { loader =>
          val definition = loader.loadClass("quotedfixture.Definitions")
          val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
          assert(KernelValidator.validate(staged).isValid, tpe)
        }
      }
    }

  test("separately compiled callers retain exact launch types and must opt in"):
    CompilerHarness.withDirectory { directory =>
      val definition = CompilerHarness.compile(directory, "Definitions", factory("if threadIdx.x < p._3 then p._2(threadIdx.x) = p._1(threadIdx.x)"))
      assertEquals(definition.errors, Vector.empty)
      val source = """
        package quotedfixture
        import scala.annotation.experimental
        import flight4s.core.ir.{DeviceBuffer, Kernel}
        class Caller:
          @experimental
          def definition: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int)] = new Definitions().definition
      """
      val caller = CompilerHarness.compile(directory, "Caller", source, dependencies = Seq(definition.classes))
      assertEquals(caller.errors, Vector.empty)
      val wrong = CompilerHarness.compile(directory, "Wrong", source.replace("DeviceBuffer[Int]", "DeviceBuffer[Float]"), dependencies = Seq(definition.classes))
      assert(wrong.errors.exists(_.contains("Required")), wrong.errors.mkString("\n"))
      val stable = CompilerHarness.compile(directory, "Stable", source.replace("@experimental", ""), dependencies = Seq(definition.classes))
      assert(stable.errors.exists(_.contains("marked @experimental")), stable.errors.mkString("\n"))
    }

  test("read-only buffers cannot be written and scalar types are checked by Scala"):
    rejected(factory("p._1(0) = 1"), "ReadOnly")
    rejected(factory("p._2(0) = 1.0f"), "Required")
    rejected(factory("if p._3 then p._2(0) = 1"), "Required")

  test("host calls values mutable captures and user-defined lookalike intrinsics are rejected"):
    rejected(factory("println(\"host\")"), "host effects")
    rejected(factory("p._2(0) = host", "val host = 7"), "captures")
    rejected(factory("p._2(0) = host", "def host: Int = 7"), "captures")
    rejected(factory("host = 2", "var host = 7"), "host mutation")
    rejected(factory("p._2(0) = threadIdx.x", "object threadIdx { def x: Int = 7 }"), "captures")
    rejected(factory("if p._3 > 0 then p._2(0) = math.abs(p._3)"), "captures")

  test("unsupported control flow definitions bindings casts and conversions fail explicitly"):
    rejected(factory("while p._3 > 0 do p._2(0) = 1"), "loops")
    rejected(factory("p._3 match { case 0 => p._2(0) = 1; case _ => () }"), "host effects")
    rejected(factory("def helper: Int = 1; p._2(0) = helper"), "local definitions")
    rejected(factory("lazy val i = threadIdx.x; p._2(i) = 1"), "cannot be lazy")
    rejected(factory("val pair = (1, 2); p._2(0) = pair._1"), "primitive locals")
    rejected(factory("p._2(0) = p._3.asInstanceOf[Int]"), "primitive operation")
    rejected(factory("p._2(0) = p._3.toFloat.toInt"), "primitive operation")
    rejected(factory("val f = (i: Int) => i + 1; p._2(0) = f(p._3)"), "primitive locals")
    rejected(factory("var alias = p._2; alias(0) = 1"), "immutable buffer aliases")

  test("expression branches remain pure and dynamic floating negation is not approximated"):
    rejected(factory("var total = 0; val result = { total = 1; total }; p._2(0) = result"), "expression blocks")
    rejected(factory("p._2(0) = if p._3 > 0 then { p._2(1) = 7; 1 } else 0"), "expression blocks")
    rejected(factory("p._2(0) = -p._3", parameters = "input[Float](\"data\"), output[Float](\"target\"), value[Float](\"scalar\")"),
      "primitive operation")

  test("unsupported existing CUDA value types are rejected at the frontend boundary"):
    rejected(factory("()", parameters = "output[flight4s.core.types.Float4](\"target\")"), "parameters only")

  test("non-literal kernel functions are rejected and markers fail outside capture"):
    rejected(factory("()", "val body: ((DeviceArray[Int, flight4s.core.ir.ReadOnly], DeviceArray[Int, flight4s.core.ir.ReadWrite], Int)) => Unit = _ => ()")
      .replace("{ p =>\n        ()\n      }", "(body)"), "literal kernel-body lambda")
    val error = intercept[IllegalStateException](ScalaKernel.threadIdx.x)
    assert(error.getMessage.contains("inside ScalaKernel.kernel"))

  test("literal-valued conditional expressions and empty signatures compile"):
    CompilerHarness.withDirectory { directory =>
      val conditional = CompilerHarness.compile(directory, "Conditional", factory("p._2(0) = if p._3 > 0 then 1 else 0"))
      assertEquals(conditional.errors, Vector.empty)
      val empty = CompilerHarness.compile(directory, "Empty", """
        import scala.annotation.experimental
        import flight4s.frontend.ScalaKernel.*
        @experimental
        def empty = kernel("empty", params()) { _ => () }
      """)
      assertEquals(empty.errors, Vector.empty)
    }

  test("intrinsic axes arithmetic comparisons integer bitwise and unary operators compile and validate"):
    val axes = for intrinsic <- Vector("threadIdx", "blockIdx", "blockDim", "gridDim"); axis <- Vector("x", "y", "z") yield s"$intrinsic.$axis"
    val operators = Vector("p._3 + 2", "p._3 - 2", "p._3 * 2", "p._3 / 2", "p._3 % 2",
      "p._3 & 2", "p._3 | 2", "p._3 ^ 2", "-p._3", "+p._3")
    val comparisons = Vector("<", "<=", ">", ">=", "==", "!=").map(op => s"if p._3 $op 2 then 1 else 0")
    val body = (axes ++ operators ++ comparisons).map(value => s"p._2(0) = $value").mkString("; ")
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Operators", factory(body))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        assertEquals(staged.body.statements.size, axes.size + operators.size + comparisons.size)
      }
    }

  test("standalone statement blocks preserve explicit device scopes"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Scope", factory("{ var total = p._3; total += 1; p._2(0) = total }; { val total = 7; p._2(1) = total }"))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        assert(staged.body.statements.forall(_.isInstanceOf[ScopedBlock]))
        assertEquals(staged.body.statements.size, 2)
      }
    }

  test("host configuration evaluates once in call order while the device body is only translated"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Configuration", """
        package quotedfixture
        import scala.annotation.experimental
        import flight4s.frontend.ScalaKernel.*
        class Definitions:
          var calls = ""
          def name(): String = { calls += "name;"; "configuration" }
          def signature() = { calls += "signature;"; params(output[Int]("target")) }
          @experimental
          def definition = kernel(name(), signature()) { p => p._1(0) = 7 }
      """)
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedfixture.Definitions")
        val instance = definition.getConstructor().newInstance()
        val staged = definition.getMethod("definition").invoke(instance).asInstanceOf[Kernel[Tuple1[DeviceBuffer[Int]]]]
        assertEquals(definition.getMethod("calls").invoke(instance), "name;signature;")
        assert(KernelValidator.validate(staged).isValid)
      }
    }
