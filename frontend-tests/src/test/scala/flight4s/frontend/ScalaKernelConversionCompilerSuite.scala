package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelConversionCompilerSuite extends FunSuite:
  private def factory(body: String, parameters: String =
      "input[Int](\"ints\"), input[Double](\"doubles\"), output[Float](\"outF\"), output[Double](\"outD\"), value[Int](\"count\")"): String = s"""
    package quotedconversionfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    class Definitions:
      @experimental
      def definition = kernel("conversions", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, parameters: String =
      "input[Int](\"ints\"), input[Double](\"doubles\"), output[Float](\"outF\"), output[Double](\"outD\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Conversions", factory(body, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedconversionfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        check(staged)
      }
    }

  private def cuda(kernel: Kernel[?]): String = CudaCodegen.generate(kernel).toOption.get.cudaSource

  test("the four exact JVM conversions stage Convert nodes with nearest-even rounding"):
    accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._5 then
        val widened = p._1(i).toDouble
        val shrunk = p._2(i).toFloat
        p._3(i) = p._1(i).toFloat + shrunk
        p._4(i) = widened + shrunk.toDouble
    """) { kernel =>
      val source = cuda(kernel)
      for marker <- Vector("::__int2double_rn(", "::__double2float_rn(", "::__int2float_rn(", "static_cast<double>(") do
        assert(source.contains(marker), marker)
    }

  test("conversions compose inside expressions and conditions"):
      accepted("""
      val i = blockIdx.x * blockDim.x + threadIdx.x
      if i < p._5 then
        val scaled = (p._1(i) * 2).toDouble
        if scaled > p._2(i) then
          p._3(i) = (p._2(i) + 0.5).toFloat
        else
          p._4(i) = (p._1(i).toFloat * 0.25f).toDouble
    """) { kernel =>
      val source = cuda(kernel)
      assert(source.contains("::__int2double_rn("))
      assert(source.contains("::__double2float_rn("))
      assert(source.contains("::__int2float_rn("))
      assert(source.contains("static_cast<double>("))
    }

  private def rejected(body: String, message: String, parameters: String =
      "input[Int](\"ints\"), input[Double](\"doubles\"), output[Float](\"outF\"), output[Double](\"outD\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedConversions", factory(body, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("float-to-int truncations stay rejected until a core policy exists"):
    rejected("val truncated = p._2(0).toInt; p._3(0) = 1.0f", "primitive operation or conversion")
    rejected("val truncated = p._3(0).toInt; p._4(0) = 1.0", "primitive operation or conversion",
      parameters = "input[Int](\"ints\"), input[Double](\"doubles\"), inOut[Float](\"outF\"), output[Double](\"outD\"), value[Int](\"count\")")
