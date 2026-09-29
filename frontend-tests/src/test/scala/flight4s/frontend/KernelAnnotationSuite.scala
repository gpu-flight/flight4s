package flight4s.frontend

import munit.FunSuite
import flight4s.core.dsl.{DslError, DslErrorCode}
import flight4s.core.ir.*

class KernelAnnotationSuite extends FunSuite:
  private def factory(body: String, setup: String = ""): String = s"""
    package frontendfixture
    import scala.annotation.experimental
    import flight4s.frontend.kernel
    import flight4s.core.dsl.CudaDsl.{kernel as buildKernel, *}
    class Definitions:
      $setup
      @experimental
      @kernel
      def definition = buildKernel("definition", params(output[Int]("target"))) { p =>
        $body
      }
  """

  private def rejected(source: String, message: String): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Rejected", source)
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
    }

  test("annotated factory snapshots a val and exposes its typed kernel to a separate opt-in caller"):
    CompilerHarness.withDirectory { directory =>
      val definition = CompilerHarness.compile(directory, "Definitions", """
        package frontendfixture
        import scala.annotation.experimental
        import flight4s.frontend.kernel
        import flight4s.core.dsl.CudaDsl.{kernel as buildKernel, *}
        class Definitions:
          @experimental
          @kernel
          def snapshots = buildKernel("snapshots", params(output[Int]("target"))) { p =>
            val old = p._1(literal(0)).read
            p._1(literal(0)) := literal(9)
            p._1(literal(1)) := old
          }
      """)
      assertEquals(definition.errors, Vector.empty)
      val caller = CompilerHarness.compile(directory, "Caller", """
        package frontendfixture
        import scala.annotation.experimental
        import flight4s.core.ir.{DeviceBuffer, Kernel}
        class Caller:
          @experimental
          def definition: Kernel[Tuple1[DeviceBuffer[Int]]] = new Definitions().snapshots
      """, experimental = false, dependencies = Seq(definition.classes))
      assertEquals(caller.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(definition.classes, caller.classes)) { loader =>
        val callerClass = loader.loadClass("frontendfixture.Caller")
        val instance = callerClass.getConstructor().newInstance()
        // Reflection is only the test boundary between separately compiled programs.
        val staged = callerClass.getMethod("definition").invoke(instance)
          .asInstanceOf[Kernel[Tuple1[DeviceBuffer[Int]]]]
        assertEquals(staged.body.statements.size, 3)
        assert(staged.body.statements.head.isInstanceOf[LocalDeclaration[?]])
        val saved = staged.body.statements.head.asInstanceOf[LocalDeclaration[Int]]
        val store = staged.body.statements.last.asInstanceOf[Store[Int, Global]]
        assertEquals(store.value, Load(saved.local, saved.span))
        assert(saved.span.file.endsWith("Definitions.scala"))
      }
    }

  test("nested statement callbacks admit ordinary snapshot vals"):
    val bodies = Vector(
      "when(threadIdx.x < literal(1)) { BODY }",
      "gpuIf(threadIdx.x < literal(1)) { BODY } { BODY }",
      "scoped { BODY }",
      "gpuFor(literal(0), literal(2)) { index => BODY }",
      "gpuRange(literal(0), literal(2)).foreach { index => BODY }",
      "gpuRange(literal(0), literal(2)).map(index => index + literal(1)).foreach { index => BODY }",
      "gpuRange(literal(0), literal(2)).map(index => (index, index + literal(1))).foreach { pair => BODY }",
      "gpuRange(literal(0), literal(2)).filter(index => index < literal(1)).foreach { index => BODY }",
      "gpuRange(literal(0), literal(2)).flatMap(index => gpuRange(literal(0), literal(2))).foreach { index => BODY }"
    )
    val snapshot = "val old = p._1(literal(0)).read; p._1(literal(0)) := literal(9); p._1(literal(1)) := old"
    bodies.foreach { body =>
      CompilerHarness.withDirectory { directory =>
        val result = CompilerHarness.compile(directory, "Nested", factory(body.replace("BODY", snapshot)))
        assertEquals(result.errors, Vector.empty, body)
      }
    }

  test("host calls are rejected instead of executing during kernel construction"):
    rejected(factory("println(\"host effect\"); p._1(literal(0)) := literal(1)"), "DSL operations")

  test("parameterless host helpers cannot bypass the call restriction"):
    rejected(factory("val i = literal(host); p._1(literal(0)) := i", "def host: Int = 7"), "DSL operations")

  test("mutable host state and external expression captures are rejected"):
    rejected(factory("val i = literal(host); p._1(literal(0)) := i", "var host = 7"), "mutable host state")
    rejected(factory("val i = external; p._1(literal(0)) := i", "val external = threadIdx.x"), "external Expr values")

  test("concrete IR node vals require the public Expr type"):
    rejected(factory("val i = flight4s.core.ir.Intrinsic(\"threadIdx.x\", flight4s.core.types.I32); p._1(literal(0)) := i"),
      "concrete IR node subtype")

  test("tuple Expr bindings require an explicit structured-state contract"):
    rejected(factory("val pair = (threadIdx.x, threadIdx.x); p._1(literal(0)) := pair._1"), "binding type")

  test("Scala control flow and mutable state remain rejected"):
    rejected(factory("if true then p._1(literal(0)) := literal(1)"), "explicit DSL control flow")
    rejected(factory("while false do p._1(literal(0)) := literal(1)"), "explicit DSL control flow")
    rejected(factory("var i = threadIdx.x; p._1(literal(0)) := i"), "var or lazy val")
    rejected(factory("lazy val i = threadIdx.x; p._1(literal(0)) := i"), "var or lazy val")

  test("expression-only traversal callbacks cannot acquire implicit statement snapshots"):
    rejected(factory("val result = gpuRange(literal(0), literal(2)).map { index => val saved = index + literal(1); saved }.sum(literal(0)); p._1(literal(0)) := result"),
      "expression-only callbacks remain pure")
    rejected(factory("gpuRange(literal(0), literal(2)).filter { index => val saved = index + literal(1); saved < literal(2) }.foreach { index => p._1(index) := index }"),
      "expression-only callbacks remain pure")
    rejected(factory("val result = gpuRange(literal(0), literal(2)).foldLeft(literal(0)) { (sum, index) => val saved = index + literal(1); sum + saved }; p._1(literal(0)) := result"),
      "expression-only callbacks remain pure")

  test("nested statement callbacks retain host capture binding and mutable-state restrictions"):
    rejected(factory("when(threadIdx.x < literal(1)) { println(\"host\") }"), "DSL operations")
    rejected(factory("scoped { val saved = external; p._1(literal(0)) := saved }", "val external = threadIdx.x"), "external Expr values")
    rejected(factory("gpuFor(literal(0), literal(2)) { index => val saved = literal(host); p._1(index) := saved }", "var host = 7"), "mutable host state")
    rejected(factory("when(threadIdx.x < literal(1)) { var saved = threadIdx.x; p._1(literal(0)) := saved }"), "var or lazy val")
    rejected(factory("scoped { val pair = (threadIdx.x, threadIdx.x); p._1(literal(0)) := pair._1 }"), "binding type")

  test("expression-only callbacks keep existing explicit statement-effect rejection"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "PureBoundary", factory(
        "val result = gpuRange(literal(0), literal(2)).map { index => scoped { val saved = index + literal(1); p._1(index) := saved }; index }.sum(literal(0)); p._1(literal(0)) := result"))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("frontendfixture.Definitions")
        val failure = intercept[java.lang.reflect.InvocationTargetException] {
          definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
        }
        val error = failure.getCause.asInstanceOf[DslError]
        assertEquals(error.code, DslErrorCode.StatementInsideExpression)
      }
    }

  test("snapshot initializers preserve compiler-created default-argument bindings"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "DefaultArgument", factory(
        "val result = gpuRange(literal(0), literal(2)).map(index => index + literal(1)).sum(literal(0)); p._1(literal(0)) := result"))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("frontendfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
          .asInstanceOf[Kernel[Tuple1[DeviceBuffer[Int]]]]
        assertEquals(staged.body.statements.size, 2)
        assert(KernelValidator.validate(staged).isValid)
      }
    }

  test("annotations reject wrong targets signatures and indirect factories"):
    val imports = """
      package frontendfixture
      import scala.annotation.experimental
      import flight4s.frontend.kernel
      import flight4s.core.dsl.CudaDsl.{kernel as buildKernel, *}
    """
    rejected(imports + "@experimental @kernel val bad = buildKernel(\"bad\") { () }", "factory methods")
    rejected(imports + "@experimental @kernel def bad(x: Int) = buildKernel(\"bad\") { () }", "no parameters")
    rejected(imports + "@experimental @kernel def bad = 1", "return Kernel")
    rejected(imports + "val existing = buildKernel(\"bad\") { () }; @experimental @kernel def bad = existing", "directly call")

  test("core callers stay stable while annotation callers require explicit opt-in"):
    CompilerHarness.withDirectory { directory =>
      val definition = CompilerHarness.compile(directory, "Definitions", factory("p._1(literal(0)) := literal(1)"))
      assertEquals(definition.errors, Vector.empty)
      val rejectedCaller = CompilerHarness.compile(directory, "RejectedCaller", """
        package frontendfixture
        class RejectedCaller:
          def kernel = new Definitions().definition
      """, dependencies = Seq(definition.classes))
      assert(rejectedCaller.errors.exists(_.contains("marked @experimental")))
      val stable = CompilerHarness.compile(directory, "StableCore", """
        import flight4s.core.dsl.CudaDsl.*
        class StableCore:
          def definition = kernel("stable") { val i = let(threadIdx.x); () }
      """)
      assertEquals(stable.errors, Vector.empty)
      val wrongArguments = CompilerHarness.compile(directory, "WrongArguments", """
        package frontendfixture
        import scala.annotation.experimental
        class WrongArguments:
          @experimental
          def invocation = new Definitions().definition.bind(Tuple1(7))
      """, dependencies = Seq(definition.classes))
      assert(wrongArguments.errors.exists(_.contains("DeviceBuffer")), wrongArguments.errors.mkString("\n"))
    }

  test("annotations need explicit experimental opt-in and input writes remain compile errors"):
    rejected(factory("p._1(literal(0)) := literal(1)").replace("@experimental", ""), "experimental")
    rejected(factory("p._1(literal(0)) := literal(1)").replace("output[Int]", "input[Int]"), "ReadWrite")
