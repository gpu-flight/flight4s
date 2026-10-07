package flight4s.frontend

import munit.FunSuite
import flight4s.core.ir.*

class ScalaKernelSharedCompilerSuite extends FunSuite:
  private def factory(body: String, parameters: String = "p", schema: String = ""): String = s"""
    package quotedsharedfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("shared", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { $parameters =>
        $body
      }
  """

  private def accepted(body: String, parameters: String = "p")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Shared", factory(body, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedsharedfixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        check(staged)
      }
    }

  test("quoted shared arrays and barriers stage existing memory and synchronization IR"):
    accepted("""
      val tile = sharedArray[Int](64)
      val lane = threadIdx.x
      val i = blockIdx.x * blockDim.x + lane
      tile(lane) = if i < p._3 then p._1(i) else 0
      barrier()
      if i < p._3 then p._2(i) = tile((lane + 1) % 64)
    """) { kernel =>
      assertEquals(kernel.ir.sharedMemory.size, 1)
      assert(kernel.ir.sharedMemory.head.size == StaticSharedMemory(64))
      assertEquals(kernel.body.statements.count(_.isInstanceOf[Barrier]), 1)
    }

  test("tupled parameters and unconditional scopes preserve typed shared aliases and unique names"):
    accepted("""
      val tile = sharedArray[Int](64)
      val alias = tile
      alias(threadIdx.x) = data(threadIdx.x)
      {
        val tile = sharedArray[Float](64)
        val doubles = sharedArray[Double](64)
        val flags = sharedArray[Boolean](64)
        tile(threadIdx.x) = 1.0f
        doubles(threadIdx.x) = 2.0
        flags(threadIdx.x) = tile(threadIdx.x) < 2.0f && doubles(threadIdx.x) == 2.0
        barrier()
        if flags(threadIdx.x) then target(threadIdx.x) = alias(threadIdx.x)
      }
    """, "(data, target, count)") { kernel =>
      assertEquals(kernel.ir.sharedMemory.size, 4)
      assertEquals(kernel.ir.sharedMemory.map(_.name).distinct.size, 4)
    }

  test("fully qualified markers aliases in loops and shared reads in pure traversals compose"):
    accepted("""
      val tile = flight4s.frontend.ScalaKernel.sharedArray[Int](64)
      tile(threadIdx.x) = 1
      flight4s.frontend.ScalaKernel.barrier()
      for round <- 0 until p._3 do
        val alias = tile
        val total = deviceRange(0, 64).map(i => alias(i)).foldLeft(0)((sum, item) => sum + item)
        barrier()
        alias(threadIdx.x) = total
        barrier()
      p._2(threadIdx.x) = tile(threadIdx.x)
    """) { kernel => assertEquals(kernel.ir.sharedMemory.size, 1) }

  private def rejected(body: String, message: String, schema: String = ""): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedShared", factory(body, schema = schema))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("sizes must be positive compile-time Int constants not device or captured values"):
    for size <- Vector("0", "-1", "p._3", "threadIdx.x", "Host.size") do
      rejected(s"val tile = sharedArray[Int]($size)", "positive compile-time Int constant", "object Host { val size = 64 }")
    rejected("val size = 64; val tile = sharedArray[Int](size)", "positive compile-time Int constant")

  test("shared declarations in conditional and repeated bodies are rejected even if unused"):
    for body <- Vector(
        "if p._3 > 0 then { val tile = sharedArray[Int](64); () }",
        "for i <- 0 until 2 do { val tile = sharedArray[Int](64); () }",
        "deviceRange(0, 2).foreach { i => val tile = sharedArray[Int](64); () }") do
      rejected(body, "outside branches and loops")

  test("shared handles must be immutable and element types primitive"):
    rejected("var tile = sharedArray[Int](64)", "must be immutable")
    rejected("val tile = sharedArray[Int](64); var alias = tile", "must be immutable")
    for tpe <- Vector("Long", "String", "(Int, Int)") do
      rejected(s"val tile = sharedArray[$tpe](64)", "Int, Float, Double and Boolean elements only")

  test("host factories captures conditional handles and lookalike API names are not erased"):
    rejected("val tile = Host.make()", "direct sharedArray declaration", "object Host { def make(): DeviceSharedArray[Int] = throw new Exception }")
    rejected("val tile = Host.tile", "direct sharedArray declaration", "object Host { val tile: DeviceSharedArray[Int] = null }")
    rejected("val tile = if p._3 > 0 then sharedArray[Int](64) else sharedArray[Int](32)", "direct sharedArray declaration")
    rejected("val tile = Host.sharedArray[Int](64)", "direct sharedArray declaration", "object Host { def sharedArray[T](size: Int): DeviceSharedArray[T] = throw new Exception }")
    rejected("Host.barrier()", "host effects", "object Host { def barrier(): Unit = println(1) }")
    rejected("val tile = Host.api.sharedArray[Int](64)", "direct sharedArray declaration", "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")
    rejected("Host.api.barrier()", "host effects", "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("shared accesses retain Scala index and value types without casts or numeric conversion"):
    rejected("val tile = sharedArray[Int](64); tile(0) = 1.0f", "Required")
    rejected("val tile = sharedArray[Int](64); p._2(0) = tile(0.0f)", "Required")
    rejected("val tile = sharedArray[Float](64); p._2(0) = tile(0)", "Required")
    rejected("val tile = sharedArray[Int](64); val global: DeviceArray[Int, flight4s.core.ir.ReadWrite] = tile", "Required")
    rejected("p._2(0) = Host.tile(0)", "declared shared array", "object Host { val tile: DeviceSharedArray[Int] = null }")
    rejected("Host.tile(0) = 1", "declared shared array", "object Host { val tile: DeviceSharedArray[Int] = null }")

  test("barriers and shared writes cannot hide in pure callbacks or expression blocks"):
    for callback <- Vector("{ barrier(); i }", "{ val tile = sharedArray[Int](64); i }") do
      rejected(s"val unused = deviceRange(0, 0).map(i => $callback)", "expression blocks")
    rejected("val tile = sharedArray[Int](64); val total = deviceRange(0, 0).foldLeft(0)((state, i) => { tile(0) = i; state })", "expression blocks")
    rejected("val result = { barrier(); 1 }; p._2(0) = result", "expression blocks")

  test("divergence diagnostics see quoted local aliases and preserve the barrier source"):
    accepted("val lane = threadIdx.x; if lane < 32 then barrier()") { kernel =>
      val warnings = KernelValidator.validate(kernel).warnings
      assertEquals(warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))
      assertNotEquals(warnings.head.span, SourceSpan.Unknown)
    }
    accepted("for i <- 0 until threadIdx.x do barrier()") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))
    }

  test("uniform condition and loop barriers retain the existing warning policy"):
    accepted("if blockIdx.x == 0 then barrier(); for i <- 0 until p._3 do barrier()") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  test("shared markers cannot allocate read write or synchronize on the JVM"):
    intercept[IllegalStateException](ScalaKernel.sharedArray[Int](64))
    intercept[IllegalStateException](ScalaKernel.barrier())
    val marker = new ScalaKernel.DeviceSharedArray[Int]()
    intercept[IllegalStateException](marker(0))
    intercept[IllegalStateException](marker(0) = 1)
