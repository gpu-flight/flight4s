package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelLocalArrayCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): String = s"""
    package quotedlocalarrayfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("localArrays", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "LocalArrays", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedlocalarrayfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        check(staged)
      }
    }

  private def all(body: Block): Vector[Stmt] = body.statements.flatMap {
    case scope: ScopedBlock => Vector(scope) ++ all(scope.body)
    case branch: IfThen => Vector(branch) ++ all(branch.thenBlock) ++ branch.elseBlock.toVector.flatMap(all)
    case loop: ForLoop => Vector(loop) ++ all(loop.body)
    case statement => Vector(statement)
  }

  test("local arrays declare once and serve static and dynamic element accesses"):
    accepted("""
      val window = localArray[Float](4)
      val alias = window
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._3 then
        window(0) = p._1(i)
        alias(1) = window(0) * 0.5f
        val slot = i % 3
        alias(slot) = alias(slot) + 2.0f
        p._2(i) = window(0) + window(1) + window(slot)
    """) { kernel =>
      val declarations = all(kernel.body).collect { case d: LocalArrayDeclaration[?] => d }
      assertEquals(declarations.size, 1)
      assertEquals(declarations.head.array.elementCount, 4)
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      assert(source.contains("[4];"), "automatic array declaration")
      assert(!source.contains("__shared__ float " + declarations.head.array.name), "local, never shared")
    }

  test("Boolean and Int elements stage and thread-local stores need no barriers"):
    accepted("""
      val flags = localArray[Boolean](2)
      val hits = localArray[Int](2)
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._3 then
        flags(0) = p._1(i) > 0.0f
        flags(1) = !flags(0)
        hits(0) = i
        hits(1) = hits(0) + 1
        p._2(i) = if flags(0) then 1.0f else 0.0f
    """) { kernel =>
      assertEquals(all(kernel.body).collect { case d: LocalArrayDeclaration[?] => d }.size, 2)
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  private def rejected(body: String, message: String, schema: String = ""): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedLocalArrays", factory(body, schema))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("sizes stay positive compile-time constants and bindings stay immutable"):
    rejected("val window = localArray[Float](p._3); p._2(0) = 1.0f", "positive compile-time Int constants")
    rejected("val window = localArray[Float](0); p._2(0) = 1.0f", "positive compile-time Int constants")
    rejected("var window = localArray[Float](4); p._2(0) = 1.0f", "must be immutable")

  test("declarations stay unconditional direct and primitive"):
    rejected("if p._3 > 0 then { val window = localArray[Float](4); p._2(0) = window(0) }", "outside branches and loops")
    rejected("val window = if p._3 > 0 then localArray[Float](4) else localArray[Float](8); p._2(0) = window(0)",
      "direct localArray declaration or immutable alias")
    rejected("val window = localArray[Long](4); p._2(0) = 1.0f", "Int, Float, Double and Boolean elements only")

  test("host calls fail on local array markers"):
    intercept[IllegalStateException](ScalaKernel.localArray[Float](4))
    val view = new ScalaKernel.DeviceLocalArray[Float]()
    intercept[IllegalStateException](view(0))
    intercept[IllegalStateException](view(0) = 1.0f)
