package flight4s.frontend

import munit.FunSuite
import flight4s.core.ir.*

class ScalaKernelSharedRankCompilerSuite extends FunSuite:
  private def factory(body: String, parameters: String = "p", schema: String = ""): String = s"""
    package quotedsharedrankfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("ranked", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { $parameters =>
        $body
      }
  """

  private def accepted(body: String, parameters: String = "p")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Ranked", factory(body, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedsharedrankfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        check(staged)
      }
    }

  test("quoted 2-D and 3-D shared arrays stage logical dimensions with stride equal to columns"):
    accepted("""
      val tile = sharedArray2D[Int](4, 32)
      val cube = sharedArray3D[Int](2, 4, 8)
      val lane = threadIdx.x
      tile(lane / 32, lane % 32) = p._1(lane)
      cube(lane % 2, lane / 32, lane % 8) = tile(lane / 32, lane % 32)
      barrier()
      p._2(lane) = tile(lane / 32, lane % 32) + cube(lane % 2, lane / 32, lane % 8)
    """) { kernel =>
      assertEquals(kernel.ir.sharedMemory.size, 2)
      assert(kernel.ir.sharedMemory(0).size == StaticSharedMemory.twoDimensional(4, 32, 32))
      assert(kernel.ir.sharedMemory(1).size == StaticSharedMemory.threeDimensional(2, 4, 8, 8))
      assertEquals(kernel.body.statements.count(_.isInstanceOf[Barrier]), 1)
    }

  test("aliases unconditional scopes and all primitive element types retain rank and unique names"):
    accepted("""
      val tile = sharedArray2D[Int](2, 32)
      val alias = tile
      alias(0, threadIdx.x) = data(threadIdx.x)
      {
        val floats = sharedArray2D[Float](2, 16)
        val doubles = sharedArray3D[Double](2, 2, 8)
        val flags = sharedArray2D[Boolean](1, 32)
        floats(1, threadIdx.x % 16) = 1.0f
        doubles(1, 1, threadIdx.x % 8) = 2.0
        flags(0, threadIdx.x) = floats(1, threadIdx.x % 16) < 2.0f && doubles(1, 1, threadIdx.x % 8) == 2.0
        barrier()
        if flags(0, threadIdx.x) then target(threadIdx.x) = alias(0, threadIdx.x)
      }
    """, "(data, target, count)") { kernel =>
      assertEquals(kernel.ir.sharedMemory.size, 4)
      assertEquals(kernel.ir.sharedMemory.map(_.name).distinct.size, 4)
      assertEquals(kernel.ir.sharedMemory.map(_.size.rank), Vector(2, 2, 3, 2))
    }

  test("rank indices compose with locals range loops and pure traversals"):
    accepted("""
      val tile = sharedArray2D[Int](4, 32)
      val lane = threadIdx.x % 32
      for row <- 0 until 4 do
        tile(row, lane) = p._1(row * 32 + lane)
      barrier()
      val total = deviceRange(0, 4).map(row => tile(row, lane)).foldLeft(0)((sum, item) => sum + item)
      p._2(lane) = total
    """) { kernel => assertEquals(kernel.ir.sharedMemory.size, 1) }

  private def rejected(body: String, message: String, schema: String = ""): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedRanked", factory(body, schema = schema))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("every dimension must be a positive compile-time Int constant not a device or captured value"):
    for size <- Vector("0", "-1", "p._3", "threadIdx.x", "Host.size") do
      rejected(s"val tile = sharedArray2D[Int]($size, 32)", "positive compile-time Int constant", "object Host { val size = 4 }")
      rejected(s"val tile = sharedArray2D[Int](4, $size)", "positive compile-time Int constant", "object Host { val size = 32 }")
      rejected(s"val cube = sharedArray3D[Int]($size, 4, 8)", "positive compile-time Int constant", "object Host { val size = 2 }")
      rejected(s"val cube = sharedArray3D[Int](2, $size, 8)", "positive compile-time Int constant", "object Host { val size = 4 }")
      rejected(s"val cube = sharedArray3D[Int](2, 4, $size)", "positive compile-time Int constant", "object Host { val size = 8 }")
    rejected("val rows = 4; val tile = sharedArray2D[Int](rows, 32)", "positive compile-time Int constant")

  test("rank-2 and rank-3 declarations in conditional and repeated bodies are rejected even if unused"):
    for body <- Vector(
        "if p._3 > 0 then { val tile = sharedArray2D[Int](2, 32); () }",
        "for i <- 0 until 2 do { val tile = sharedArray2D[Int](2, 32); () }",
        "deviceRange(0, 2).foreach { i => val cube = sharedArray3D[Int](2, 2, 8); () }") do
      rejected(body, "outside branches and loops")

  test("rank-2 and rank-3 handles must be immutable and element types primitive"):
    rejected("var tile = sharedArray2D[Int](2, 32)", "must be immutable")
    rejected("val tile = sharedArray2D[Int](2, 32); var alias = tile", "must be immutable")
    rejected("var cube = sharedArray3D[Int](2, 2, 8)", "must be immutable")
    for tpe <- Vector("Long", "String", "(Int, Int)") do
      rejected(s"val tile = sharedArray2D[$tpe](2, 32)", "Int, Float, Double and Boolean elements only")
      rejected(s"val cube = sharedArray3D[$tpe](2, 2, 8)", "Int, Float, Double and Boolean elements only")

  test("host factories captures conditional handles and lookalike rank APIs are not erased"):
    rejected("val tile = Host.make()", "direct sharedArray2D declaration", "object Host { def make(): DeviceSharedArray2D[Int] = throw new Exception }")
    rejected("val cube = Host.cube", "direct sharedArray3D declaration", "object Host { val cube: DeviceSharedArray3D[Int] = null }")
    rejected("val tile = if p._3 > 0 then sharedArray2D[Int](2, 32) else sharedArray2D[Int](2, 16)", "direct sharedArray2D declaration")
    rejected("val tile = Host.sharedArray2D[Int](2, 32)", "direct sharedArray2D declaration", "object Host { def sharedArray2D[T](rows: Int, columns: Int): DeviceSharedArray2D[T] = throw new Exception }")
    rejected("val cube = Host.api.sharedArray3D[Int](2, 2, 8)", "direct sharedArray3D declaration", "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("rank accesses retain Scala index arity and value types without casts or numeric conversion"):
    rejected("val tile = sharedArray2D[Int](2, 32); tile(0, 0) = 1.0f", "Required")
    rejected("val tile = sharedArray2D[Int](2, 32); p._2(0) = tile(0.0f, 0)", "Required")
    rejected("val tile = sharedArray2D[Float](2, 32); p._2(0) = tile(0, 0)", "Required")
    rejected("val tile = sharedArray2D[Int](2, 32); p._2(0) = tile(0)", "missing argument")
    rejected("val cube = sharedArray3D[Int](2, 2, 8); cube(0, 0) = 1", "missing argument")
    rejected("p._2(0) = Host.tile(0, 0)", "declared shared array", "object Host { val tile: DeviceSharedArray2D[Int] = null }")
    rejected("Host.cube(0, 0, 0) = 1", "declared shared array", "object Host { val cube: DeviceSharedArray3D[Int] = null }")

  test("rank declarations and writes cannot hide in pure callbacks or expression blocks"):
    for callback <- Vector("{ val tile = sharedArray2D[Int](2, 32); i }", "{ val cube = sharedArray3D[Int](2, 2, 8); i }") do
      rejected(s"val unused = deviceRange(0, 0).map(i => $callback)", "expression blocks")
    rejected("val tile = sharedArray2D[Int](2, 32); val total = deviceRange(0, 0).foldLeft(0)((state, i) => { tile(0, 0) = i; state })", "expression blocks")

  test("rank markers cannot allocate read or write on the JVM"):
    intercept[IllegalStateException](ScalaKernel.sharedArray2D[Int](2, 32))
    intercept[IllegalStateException](ScalaKernel.sharedArray3D[Int](2, 2, 8))
    val tile = new ScalaKernel.DeviceSharedArray2D[Int]()
    intercept[IllegalStateException](tile(0, 0))
    intercept[IllegalStateException](tile(0, 0) = 1)
    val cube = new ScalaKernel.DeviceSharedArray3D[Int]()
    intercept[IllegalStateException](cube(0, 0, 0))
    intercept[IllegalStateException](cube(0, 0, 0) = 1)
