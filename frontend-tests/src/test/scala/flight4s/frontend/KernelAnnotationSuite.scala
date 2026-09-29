package flight4s.frontend

import munit.FunSuite
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

  test("Scala control flow mutable state and nested implicit snapshots are rejected"):
    rejected(factory("if true then p._1(literal(0)) := literal(1)"), "explicit DSL control flow")
    rejected(factory("while false do p._1(literal(0)) := literal(1)"), "explicit DSL control flow")
    rejected(factory("var i = threadIdx.x; p._1(literal(0)) := i"), "var or lazy val")
    rejected(factory("lazy val i = threadIdx.x; p._1(literal(0)) := i"), "var or lazy val")
    rejected(factory("when(threadIdx.x < literal(1)) { val i = threadIdx.x; p._1(literal(0)) := i }"),
      "top-level Expr vals")

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
