package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelDeviceMathCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Float](\"bias\"), value[Int](\"count\")"): String = s"""
    package quoteddevicemathfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("deviceMath", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Float](\"bias\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "DeviceMath", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quoteddevicemathfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        check(staged)
      }
    }

  private def cuda(kernel: Kernel[?]): String = CudaCodegen.generate(kernel).toOption.get.cudaSource

  test("all five markers stage Float libm calls with the f suffix"):
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._4 then
        val grown = exp(p._1(i))
        val shrunk = log(grown + 1.0f)
        val root = sqrt(shrunk + 2.0f)
        val inverse = rsqrt(root + 3.0f)
        p._2(i) = tanh(inverse + p._3)
    """) { kernel =>
      val source = cuda(kernel)
      for marker <- Vector("::expf(", "::logf(", "::sqrtf(", "::rsqrtf(", "::tanhf(") do
        assert(source.contains(marker), marker)
      assert(!source.contains("fast_math"))
    }

  test("Double elements emit unsuffixed libm calls composed inside larger expressions"):
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._4 then
        val smooth = tanh(exp(p._1(i)) - log(p._1(i) * p._1(i) + 1.0))
        p._2(i) = if sqrt(smooth + 2.0) > p._3 then smooth else rsqrt(smooth + 3.0) * 0.5
    """, parameters = "input[Double](\"data\"), output[Double](\"target\"), value[Double](\"bias\"), value[Int](\"count\")") { kernel =>
      val source = cuda(kernel)
      for marker <- Vector("::exp(", "::log(", "::sqrt(", "::rsqrt(", "::tanh(") do
        assert(source.contains(marker), marker)
      for suffixed <- Vector("::expf(", "::logf(", "::sqrtf(", "::rsqrtf(", "::tanhf(") do
        assert(!source.contains(suffixed), suffixed)
    }

  test("bare renamed and qualified markers share the same lowering"):
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._4 then
        p._2(i) = squash(flight4s.frontend.ScalaKernel.sqrt(p._1(i) * p._1(i) + 1.0f))
    """, schema = "import flight4s.frontend.ScalaKernel.{tanh as squash}") { kernel =>
      val source = cuda(kernel)
      assert(source.contains("::tanhf("))
      assert(source.contains("::sqrtf("))
    }

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Float](\"bias\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedDeviceMath", factory(body, schema, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("Int and Boolean values reject and host math never stages"):
    rejected("val grown = exp(p._4); p._2(0) = 1.0f", "Float and Double values only")
    rejected("val flag = tanh(p._4 > 0); p._2(0) = 1.0f", "Float and Double values only")
    rejected("val host = scala.math.sqrt(p._3 + 1.0); p._2(0) = 1.0f", "does not translate this expression",
      parameters = "input[Double](\"data\"), output[Float](\"target\"), value[Double](\"bias\"), value[Int](\"count\")")

  test("host calls fail on device math markers"):
    intercept[IllegalStateException](ScalaKernel.exp(1.0f))
    intercept[IllegalStateException](ScalaKernel.log(1.0f))
    intercept[IllegalStateException](ScalaKernel.sqrt(1.0))
    intercept[IllegalStateException](ScalaKernel.rsqrt(1.0))
    intercept[IllegalStateException](ScalaKernel.tanh(0.5f))
