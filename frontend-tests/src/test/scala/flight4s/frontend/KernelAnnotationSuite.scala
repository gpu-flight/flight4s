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

  test("mutable Expr locals lower to device declarations reads and stores"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "MutableLocals", factory(
        "var total = literal(1); val before = total; total = total + literal(2); p._1(literal(0)) := before; p._1(literal(1)) := total"))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("frontendfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
          .asInstanceOf[Kernel[Tuple1[DeviceBuffer[Int]]]]
        assertEquals(staged.body.statements.size, 5)
        val total = staged.body.statements(0).asInstanceOf[LocalDeclaration[Int]]
        val before = staged.body.statements(1).asInstanceOf[LocalDeclaration[Int]]
        val assignment = staged.body.statements(2).asInstanceOf[Store[Int, Local]]
        assertEquals(assignment.to, total.local)
        assertEquals(before.initial.asInstanceOf[Load[Int, Local, ReadWrite]].from, total.local)
        assertEquals(staged.body.statements(3).asInstanceOf[Store[Int, Global]].value
          .asInstanceOf[Load[Int, Local, ReadWrite]].from, before.local)
        assertEquals(staged.body.statements(4).asInstanceOf[Store[Int, Global]].value
          .asInstanceOf[Load[Int, Local, ReadWrite]].from, total.local)
        assert(KernelValidator.validate(staged).isValid)
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

  test("Scala control flow host vars and lazy state remain rejected"):
    rejected(factory("if true then p._1(literal(0)) := literal(1)"), "explicit DSL control flow")
    rejected(factory("while false do p._1(literal(0)) := literal(1)"), "explicit DSL control flow")
    rejected(factory("var i = 0; p._1(literal(0)) := literal(i)"), "host vars are not translated")
    rejected(factory("lazy val i = threadIdx.x; p._1(literal(0)) := i"), "lazy val")

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
    rejected(factory("when(threadIdx.x < literal(1)) { var saved = 0; p._1(literal(0)) := literal(saved) }"), "host vars are not translated")
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

  test("nested mutable locals capture outer device state and preserve same-name shadowing"):
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
    bodies.foreach { body =>
      CompilerHarness.withDirectory { directory =>
        val edits = "total += literal(2); var inner = total; val saved = inner; inner = inner + literal(3); scoped { var inner = literal(4); inner = inner + literal(1); p._1(literal(0)) := inner }; total = inner; p._1(literal(1)) := saved"
        val result = CompilerHarness.compile(directory, "NestedMutable", factory(
          "var total = literal(1); " + body.replace("BODY", edits) + "; p._1(literal(2)) := total"))
        assertEquals(result.errors, Vector.empty, body)
        CompilerHarness.withClasses(Seq(result.classes)) { loader =>
          val definition = loader.loadClass("frontendfixture.Definitions")
          val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
            .asInstanceOf[Kernel[Tuple1[DeviceBuffer[Int]]]]
          assert(KernelValidator.validate(staged).isValid, body)
        }
      }
    }

  test("mutable locals keep exact Expr types and reject host assignments and expression-block effects"):
    rejected(factory("var total = flight4s.core.ir.Intrinsic(\"threadIdx.x\", flight4s.core.types.I32); p._1(literal(0)) := total"),
      "concrete IR node subtype")
    rejected(factory("var pair = (threadIdx.x, threadIdx.x); p._1(literal(0)) := pair._1"), "initialized Expr[T]")
    rejected(factory("host = 2; p._1(literal(0)) := literal(1)", "var host = 7"), "host mutation")
    rejected(factory("var total = literal(0); val result = { total = literal(1); total }; p._1(literal(0)) := result"),
      "statement-producing DSL body")

  test("pure traversal callbacks reject mutable declarations and captured device assignments"):
    rejected(factory("val result = gpuRange(literal(0), literal(2)).map { index => var total = index; total }.sum(literal(0)); p._1(literal(0)) := result"),
      "statement-producing DSL body")
    rejected(factory("var total = literal(0); val result = gpuRange(literal(0), literal(2)).map { index => total = total + index; index }.sum(literal(0)); p._1(literal(0)) := result"),
      "expression-only callbacks remain pure")
    rejected(factory("var total = literal(0); gpuRange(literal(0), literal(2)).filter { index => total = total + index; index < literal(1) }.foreach { index => p._1(index) := index }"),
      "expression-only callbacks remain pure")
    rejected(factory("var total = literal(0); val result = gpuRange(literal(0), literal(2)).foldLeft(literal(0)) { (sum, index) => total = total + index; sum + index }; p._1(literal(0)) := result"),
      "expression-only callbacks remain pure")

  test("nested device assignment cannot bypass expression staging through a scoped callback"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "MutablePureBoundary", factory(
        "var total = literal(0); val result = gpuRange(literal(0), literal(2)).map { index => scoped { total = total + index }; index }.sum(literal(0)); p._1(literal(0)) := result"))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("frontendfixture.Definitions")
        val failure = intercept[java.lang.reflect.InvocationTargetException] {
          definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
        }
        assertEquals(failure.getCause.asInstanceOf[DslError].code, DslErrorCode.StatementInsideExpression)
      }
    }

  test("mutable initialization retains default getters and device reads in pure callbacks"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "MutableDefaultArgument", factory(
        "var total = gpuRange(literal(0), literal(2)).map(index => index + literal(1)).sum(literal(0)); total += literal(1); val result = gpuRange(literal(0), literal(2)).map(index => total + index).sum(literal(0)); p._1(literal(0)) := result"))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("frontendfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
          .asInstanceOf[Kernel[Tuple1[DeviceBuffer[Int]]]]
        assertEquals(staged.body.statements.size, 4)
        assert(KernelValidator.validate(staged).isValid)
      }
    }

  test("mutable lowering preserves scalar and native vector device types"):
    val shapes = Vector(
      ("Int", "literal(1)"),
      ("Float", "literal(1.0f)"),
      ("Double", "literal(1.0)"),
      ("Boolean", "threadIdx.x < literal(1)"),
      ("flight4s.core.types.UInt", "literal(flight4s.core.types.UInt.fromBits(1))"),
      ("flight4s.core.types.Float16", "literal(flight4s.core.types.Float16.fromBits(0))"),
      ("flight4s.core.types.BFloat16", "literal(flight4s.core.types.BFloat16.fromBits(0))"),
      ("flight4s.core.types.Float8E4M3", "literal(flight4s.core.types.Float8E4M3.fromBits(0))"),
      ("flight4s.core.types.Float8E5M2", "literal(flight4s.core.types.Float8E5M2.fromBits(0))"),
      ("flight4s.core.types.Float2", "float2(literal(1.0f), literal(2.0f))"),
      ("flight4s.core.types.Float4", "float4(literal(1.0f), literal(2.0f), literal(3.0f), literal(4.0f))"))
    shapes.foreach { (valueType, initial) =>
      CompilerHarness.withDirectory { directory =>
        val source = factory(s"var total = $initial; total = total; p._1(literal(0)) := total")
          .replace("output[Int]", s"output[$valueType]")
        val result = CompilerHarness.compile(directory, "MutableType", source)
        assertEquals(result.errors, Vector.empty, valueType)
        CompilerHarness.withClasses(Seq(result.classes)) { loader =>
          val definition = loader.loadClass("frontendfixture.Definitions")
          val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
            .asInstanceOf[Kernel[?]]
          assert(KernelValidator.validate(staged).isValid, valueType)
        }
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
