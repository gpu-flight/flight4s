package flight4s.frontend

import munit.FunSuite

/** One probe per row of the frontend README's boundary table, so the
  * documented diagnostics and the macro's actual messages cannot drift. */
class ScalaKernelDiagnosticsSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): String = s"""
    package quoteddiagnosticsfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    import flight4s.core.ir.MemoryOrder
    $schema
    class Definitions:
      @experimental
      def definition = kernel("diagnostics", params($parameters)) { p =>
        $body
      }
  """

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Diagnostics", factory(body, schema, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  private val intParameters = "input[Int](\"data\"), inOut[Int](\"cells\"), value[Int](\"count\")"

  test("control flow and binding boundaries reject with their documented diagnostics"):
    rejected("for k <- 0 to 3 do p._2(k) = 1.0f", "direct start until end with unit stride")
    rejected("lazy val missing = 1; p._2(0) = 1.0f", "must be initialized and cannot be lazy")
    rejected("def helper(x: Int): Int = x; p._2(0) = 1.0f", "local definitions or imports")
    rejected("p._2(0) = { p._2(1) = 2.0f; 1.0f }", "expression blocks")

  test("host value and call boundaries reject with their documented diagnostics"):
    rejected("val unused = deviceRange(0, p._3).map(i => (i, host)); p._2(0) = 1.0f", "captures", "val host = 7")
    rejected("val x = heat(); p._2(0) = x", "does not translate this expression", "def heat(): Float = 1.0f")

  test("conversion and shift boundaries reject with their documented diagnostics"):
    rejected("val truncated = p._1(0).toInt; p._2(0) = 1.0f", "primitive operation or conversion")
    rejected("p._2(0) = p._1(0) << 2L", "primitive operation or conversion", parameters = intParameters)

  test("collective and atomic boundaries reject with their documented diagnostics"):
    rejected("val s = warpReduceSum(p._3, 1.0f); p._2(0) = s", "compile-time constant Int values")
    rejected("atomicDevice(p._2(0)).store(1, if p._3 > 0 then MemoryOrder.Relaxed else MemoryOrder.Release)",
      "direct MemoryOrder case references", parameters = intParameters)
    rejected("val seen = atomicDevice(p._2(0)).compareExchange(0, 1, MemoryOrder.Relaxed, MemoryOrder.Acquire); p._2(1) = seen",
      "failure orders", parameters = intParameters)
    rejected("val low = atomicDevice(p._2(0)).fetchMin(0.5f, MemoryOrder.Relaxed); p._2(1) = low",
      "Int elements only", parameters = "input[Float](\"data\"), inOut[Float](\"cells\"), value[Int](\"count\")")
    rejected("p._2(0) = atomicDevice(p._2(1)).fetchAdd(1, MemoryOrder.Relaxed)", "directly initialize",
      parameters = intParameters)

  test("storage boundaries reject with their documented diagnostics"):
    rejected("val window = localArray[Float](p._3); p._2(0) = 1.0f", "positive compile-time Int constants")
    rejected("if p._3 > 0 then { val tile = sharedArray[Float](4); p._2(0) = tile(0) }", "outside branches and loops")
    rejected("val window = localArray[Long](2); p._2(0) = 1.0f", "Int, Float, Double and Boolean")
