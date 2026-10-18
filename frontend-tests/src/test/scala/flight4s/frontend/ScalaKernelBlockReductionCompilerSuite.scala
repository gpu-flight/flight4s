package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelBlockReductionCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")", block: Int = 4): String = s"""
    package quotedblockreducefixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    import flight4s.core.launch.{Block as LaunchBlock}
    $schema
    class Definitions:
      @experimental
      def definition = kernel("reduce", params($parameters)) { p =>
        $body
      }.requiringBlock(LaunchBlock.x($block))
  """

  private def staged(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")", block: Int = 4)(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "BlockReduce", factory(body, schema, parameters, block))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedblockreducefixture.Definitions")
        val kernel = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        check(kernel)
      }
    }

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")", block: Int = 4)(check: Kernel[Tuple] => Unit): Unit =
    staged(body, schema, parameters, block) { kernel =>
      assert(KernelValidator.validate(kernel).isValid)
      check(kernel)
    }

  private def all(body: Block): Vector[Stmt] = body.statements.flatMap {
    case scope: ScopedBlock => Vector(scope) ++ all(scope.body)
    case branch: IfThen => Vector(branch) ++ all(branch.thenBlock) ++ branch.elseBlock.toVector.flatMap(all)
    case loop: ForLoop => Vector(loop) ++ all(loop.body)
    case statement => Vector(statement)
  }

  private def syncthreadsCount(kernel: Kernel[?]): Int =
    val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
    source.sliding("__syncthreads();".length).count(_ == "__syncthreads();")

  test("block sums stage scratch one result local and a barriered strided tree"):
    accepted("val scratch = blockReduction[Float](4); val total = scratch.sum(p._1(threadIdx.x)); p._2(threadIdx.x) = total") { kernel =>
      assertEquals(kernel.sharedMemory.size, 1)
      // One store barrier, one per tree level, one trailing reuse barrier.
      assertEquals(all(kernel.body).count(_.isInstanceOf[Barrier]), 4)
      assertEquals(syncthreadsCount(kernel), 4)
      assert(all(kernel.body).exists(_.isInstanceOf[ScopedBlock]))
      assert(all(kernel.body).forall(_.span != SourceSpan.Unknown))
      assertEquals(kernel.ir.blockRequirements.size, 1)
    }

  test("explicit combines reuse one scratch through aliases with matching element types"):
    accepted("""
      val scratch = blockReduction[Int](8)
      val alias = scratch
      val item = p._1(threadIdx.x)
      val best = scratch.reduceTree(item)((a, b) => if a > b then a else b)
      val total = alias.sum(item)
      p._2(threadIdx.x) = best + total
    """, parameters = "input[Int](\"data\"), output[Int](\"target\"), value[Int](\"count\")", block = 8) { kernel =>
      assertEquals(kernel.sharedMemory.size, 1)
      // Two calls on one scratch: each stages 1 + log2(8) + 1 barriers.
      assertEquals(all(kernel.body).count(_.isInstanceOf[Barrier]), 10)
      assertEquals(kernel.ir.blockRequirements.size, 2)
    }

  test("Double reductions inside uniform loops restage per call"):
    accepted("""
      val scratch = blockReduction[Double](4)
      var total = 0.0
      for round <- deviceRange(0, p._3) do
        val sum = scratch.sum(p._1(threadIdx.x))
        total += sum
      p._2(threadIdx.x) = total
    """, parameters = "input[Double](\"data\"), output[Double](\"target\"), value[Int](\"count\")") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[Barrier]), 4)
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  test("kernels must require exactly the declared block shape"):
    staged("val scratch = blockReduction[Float](8); val total = scratch.sum(p._1(0)); p._2(0) = total", block = 4) { kernel =>
      val validation = KernelValidator.validate(kernel)
      assert(!validation.isValid)
      assert(validation.errors.exists(_.message.contains("block collective requires")))
    }

  test("divergent control warns per staged barrier while block-uniform control stays quiet"):
    accepted("val scratch = blockReduction[Float](4); if threadIdx.x < p._3 then { val total = scratch.sum(p._1(0)); p._2(0) = total }") { kernel =>
      val warnings = KernelValidator.validate(kernel).warnings
      assertEquals(warnings.map(_.code), Vector.fill(4)(ValidationWarningCode.BarrierMayDiverge))
    }
    accepted("val scratch = blockReduction[Float](4); if blockIdx.x == 0 then { val total = scratch.sum(p._1(0)); p._2(0) = total }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")", block: Int = 4): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedBlockReduce", factory(body, schema, parameters, block))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("thread counts must be compile-time powers of two from 1 through 1024"):
    rejected("val scratch = blockReduction[Float](0); ()", "power-of-two")
    rejected("val scratch = blockReduction[Float](3); ()", "power-of-two")
    rejected("val scratch = blockReduction[Float](2048); ()", "power-of-two")
    rejected("val scratch = blockReduction[Float](p._3); ()", "power-of-two")

  test("declarations stay immutable unconditional typed and directly declared"):
    rejected("var scratch = blockReduction[Float](4); ()", "must be immutable")
    rejected("if p._3 > 0 then { val scratch = blockReduction[Float](4); () }", "outside branches and loops")
    rejected("for i <- 0 until 2 do { val scratch = blockReduction[Float](4); () }", "outside branches and loops")
    rejected("val scratch = blockReduction[Boolean](4); ()", "Int, Float and Double")
    rejected("val scratch = Host.api.blockReduction[Float](4); ()", "direct blockReduction declaration or immutable alias",
      "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("results are initializer-only and combines stay pure literal lambdas"):
    rejected("val scratch = blockReduction[Float](4); p._2(0) = scratch.sum(p._1(0))", "directly initialize")
    rejected("val scratch = blockReduction[Float](4); val total = scratch.sum(p._1(0)) + 1.0f; p._2(0) = total", "directly initialize")
    rejected("val scratch = blockReduction[Float](4); val total = scratch.reduceTree(p._1(0))((a, b) => { p._2(0) = a; a + b }); p._2(0) = total", "expression blocks")
    rejected("val scratch = blockReduction[Float](4); val f = (a: Float, b: Float) => a + b; val total = scratch.reduceTree(p._1(0))(f); p._2(0) = total", "primitive locals")

  test("host calls fail on block reduction markers"):
    intercept[IllegalStateException](ScalaKernel.blockReduction[Float](32))
    val scratch = new ScalaKernel.DeviceBlockReduction[Float]()
    intercept[IllegalStateException](scratch.sum(1.0f))
    intercept[IllegalStateException](scratch.reduceTree(1.0f)(_ + _))
