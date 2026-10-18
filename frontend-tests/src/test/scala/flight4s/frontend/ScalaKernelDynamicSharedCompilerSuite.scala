package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, DynamicSharedMemoryRequirement}
import flight4s.core.ir.*

class ScalaKernelDynamicSharedCompilerSuite extends FunSuite:
  private def factory(body: String, parameters: String = "p", schema: String = ""): String = s"""
    package quoteddynamicfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("dynamic", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { $parameters =>
        $body
      }
  """

  private def staged(body: String, parameters: String = "p")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Dynamic", factory(body, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quoteddynamicfixture.Definitions")
        val kernel = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        check(kernel)
      }
    }

  private def accepted(body: String, parameters: String = "p")(check: Kernel[Tuple] => Unit): Unit =
    staged(body, parameters) { kernel =>
      assert(KernelValidator.validate(kernel).isValid)
      check(kernel)
    }

  test("quoted dynamic shared arrays stage sizeless storage and record the launch requirement"):
    accepted("""
      val tile = dynamicSharedArray[Int]()
      val lane = threadIdx.x
      tile(lane) = if lane < p._3 then p._1(lane) else 0
      barrier()
      if lane < p._3 then p._2(lane) = tile((lane + 1) % 64)
    """) { kernel =>
      assertEquals(kernel.ir.sharedMemory.size, 1)
      assert(kernel.ir.sharedMemory.head.size == DynamicSharedMemory)
      assertEquals(kernel.body.statements.count(_.isInstanceOf[Barrier]), 1)
      assertEquals(CudaCodegen.generate(kernel).toOption.get.launchRequirements.dynamicSharedMemory,
        Some(DynamicSharedMemoryRequirement(elementSizeBytes = 4, elementAlignmentBytes = 4)))
    }

  test("dynamic handles alias and coexist with static shared arrays under unique names"):
    accepted("""
      val tile = dynamicSharedArray[Double]()
      val alias = tile
      val fixed = sharedArray[Double](32)
      val lane = threadIdx.x
      alias(lane) = if lane < p._3 then 1.0 else 0.0
      fixed(lane % 32) = alias(lane)
      barrier()
      if alias(lane) < fixed(lane % 32) then p._2(lane) = 1
    """) { kernel =>
      assertEquals(kernel.ir.sharedMemory.size, 2)
      assertEquals(kernel.ir.sharedMemory.map(_.name).distinct.size, 2)
      assert(kernel.ir.sharedMemory.head.size == DynamicSharedMemory)
      assert(kernel.ir.sharedMemory(1).size == StaticSharedMemory(32))
      assertEquals(CudaCodegen.generate(kernel).toOption.get.launchRequirements.dynamicSharedMemory,
        Some(DynamicSharedMemoryRequirement(elementSizeBytes = 8, elementAlignmentBytes = 8)))
    }

  test("a second dynamic declaration stages but existing validation rejects the kernel"):
    staged("""
      val one = dynamicSharedArray[Int]()
      val two = dynamicSharedArray[Float]()
      one(threadIdx.x) = 1
      two(threadIdx.x) = 1.0f
      p._2(0) = one(0)
    """) { kernel =>
      assertEquals(KernelValidator.validate(kernel).errors.map(_.code),
        Vector(ValidationCode.MultipleDynamicSharedDeclarations))
    }

  private def rejected(body: String, message: String, schema: String = ""): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedDynamic", factory(body, schema = schema))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("dynamic declarations take no size and the declaration site stays constrained"):
    rejected("val tile = dynamicSharedArray[Int](64)", "arguments")
    rejected("if p._3 > 0 then { val tile = dynamicSharedArray[Int](); () }", "outside branches and loops")
    rejected("for i <- 0 until 2 do { val tile = dynamicSharedArray[Int](); () }", "outside branches and loops")
    rejected("deviceRange(0, 2).foreach { i => val tile = dynamicSharedArray[Int](); () }", "outside branches and loops")

  test("dynamic handles must be immutable and element types primitive"):
    rejected("var tile = dynamicSharedArray[Int]()", "must be immutable")
    rejected("val tile = dynamicSharedArray[Int](); var alias = tile", "must be immutable")
    for tpe <- Vector("Long", "String", "(Int, Int)") do
      rejected(s"val tile = dynamicSharedArray[$tpe]()", "Int, Float, Double and Boolean elements only")

  test("host factories captures conditional handles and lookalike dynamic APIs are not erased"):
    rejected("val tile = Host.make()", "direct dynamicSharedArray declaration", "object Host { def make(): DeviceDynamicSharedArray[Int] = throw new Exception }")
    rejected("val tile = Host.tile", "direct dynamicSharedArray declaration", "object Host { val tile: DeviceDynamicSharedArray[Int] = null }")
    rejected("val tile = if p._3 > 0 then dynamicSharedArray[Int]() else dynamicSharedArray[Int]()", "direct dynamicSharedArray declaration")
    rejected("val tile = Host.dynamicSharedArray[Int]()", "direct dynamicSharedArray declaration", "object Host { def dynamicSharedArray[T](): DeviceDynamicSharedArray[T] = throw new Exception }")
    rejected("val tile = Host.api.dynamicSharedArray[Int]()", "direct dynamicSharedArray declaration", "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("dynamic accesses retain Scala index and value types and reference declared storage"):
    rejected("val tile = dynamicSharedArray[Int](); tile(0) = 1.0f", "Required")
    rejected("val tile = dynamicSharedArray[Int](); p._2(0) = tile(0.0f)", "Required")
    rejected("val tile = dynamicSharedArray[Float](); p._2(0) = tile(0)", "Required")
    rejected("p._2(0) = Host.tile(0)", "declared shared array", "object Host { val tile: DeviceDynamicSharedArray[Int] = null }")
    rejected("Host.tile(0) = 1", "declared shared array", "object Host { val tile: DeviceDynamicSharedArray[Int] = null }")

  test("dynamic declarations and writes cannot hide in pure callbacks or expression blocks"):
    rejected("val unused = deviceRange(0, 0).map(i => { val tile = dynamicSharedArray[Int](); i })", "expression blocks")
    rejected("val tile = dynamicSharedArray[Int](); val total = deviceRange(0, 0).foldLeft(0)((state, i) => { tile(0) = i; state })", "expression blocks")

  test("dynamic markers cannot allocate read or write on the JVM"):
    intercept[IllegalStateException](ScalaKernel.dynamicSharedArray[Int]())
    val tile = new ScalaKernel.DeviceDynamicSharedArray[Int]()
    intercept[IllegalStateException](tile(0))
    intercept[IllegalStateException](tile(0) = 1)
