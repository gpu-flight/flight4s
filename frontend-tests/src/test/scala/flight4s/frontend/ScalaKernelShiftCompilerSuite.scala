package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelShiftCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), output[Int](\"target\"), value[Int](\"bias\"), value[Int](\"count\")"): String = s"""
    package quotedshiftfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("shifts", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), output[Int](\"target\"), value[Int](\"bias\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Shifts", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedshiftfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        check(staged)
      }
    }

  private def cuda(kernel: Kernel[?]): String = CudaCodegen.generate(kernel).toOption.get.cudaSource

  test("all three Int shifts stage the UB-free signed-shift emission with JVM masking"):
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._4 then
        val rotated = (p._1(i) << p._3) | (p._1(i) >>> (32 - p._3))
        val arithmetic = p._1(i) >> 2
        p._2(i) = rotated ^ arithmetic
    """) { kernel =>
      val source = cuda(kernel)
      assert(source.contains("& 31u"), "distance masking")
      assert(source.contains("0x7fffffffu"), "bounded cast back")
      assert(source.contains("0x80000000u"), "arithmetic right sign handling")
    }

  test("bitwise complement stages the explicit DSL all-bits XOR"):
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._4 then
        p._2(i) = ~(p._1(i) & p._3)
    """) { kernel =>
      val source = cuda(kernel)
      assert(source.contains("^ -1"), "complement as XOR with all bits set")
    }

  test("Float and Double unary minus stage exact IEEE sign flips"):
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._4 then
        p._2(i) = -p._1(i) * p._3 + -(p._1(i) - p._3)
    """, parameters = "input[Float](\"data\"), output[Float](\"target\"), value[Float](\"bias\"), value[Int](\"count\")") { kernel =>
      val source = cuda(kernel)
      assert(source.contains("(-0x1.0p0f * "), "Float negation multiplies by -1.0f")
    }
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._4 then
        p._2(i) = -(p._1(i) * p._1(i))
    """, parameters = "input[Double](\"data\"), output[Double](\"target\"), value[Double](\"bias\"), value[Int](\"count\")") { kernel =>
      val source = cuda(kernel)
      assert(source.contains("(-0x1.0p0 * "), "Double negation multiplies by -1.0")
    }

  test("Int unary minus keeps its zero-subtraction lowering"):
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._4 then
        p._2(i) = -p._1(i)
    """) { kernel =>
      val source = cuda(kernel)
      assert(source.contains("0 - "), "Int negation stays subtract from zero")
    }

  private def rejected(body: String, message: String, parameters: String =
      "input[Int](\"data\"), output[Int](\"target\"), value[Int](\"bias\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedShifts", factory(body, "", parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("Long shift distances reject at the existing conversion boundary"):
    rejected("p._2(0) = p._1(0) << 2L", "primitive operation or conversion")
    rejected("p._2(0) = p._1(0) >> 2L", "primitive operation or conversion")
    rejected("p._2(0) = p._1(0) >>> 2L", "primitive operation or conversion")
