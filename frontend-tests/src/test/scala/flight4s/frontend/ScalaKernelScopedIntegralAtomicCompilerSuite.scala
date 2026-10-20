package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelScopedIntegralAtomicCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"cells\"), value[Int](\"count\")"): String = s"""
    package quotedscopedintegralfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    import flight4s.core.ir.MemoryOrder
    $schema
    class Definitions:
      @experimental
      def definition = kernel("scopedIntegral", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"cells\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "ScopedIntegralAtomics", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedscopedintegralfixture.Definitions")
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

  private def results(kernel: Kernel[?]): Vector[AtomicResult[?, ?]] =
    all(kernel.body).collect { case result: AtomicResult[?, ?] => result }

  test("all five integral operations and compareExchange stage with device scope and orders"):
    accepted("""
      val low = atomicDevice(p._2(0)).fetchMin(p._1(0), MemoryOrder.Relaxed)
      val high = atomicDevice(p._2(1)).fetchMax(low, MemoryOrder.Acquire)
      val masked = atomicDevice(p._2(2)).fetchAnd(high, MemoryOrder.Release)
      val merged = atomicDevice(p._2(3)).fetchOr(masked, MemoryOrder.AcquireRelease)
      val flipped = atomicDevice(p._2(4)).fetchXor(merged, MemoryOrder.SequentiallyConsistent)
      val observed = atomicDevice(p._2(5)).compareExchange(flipped, merged, MemoryOrder.AcquireRelease, MemoryOrder.Acquire)
      p._2(6) = observed
    """) { kernel =>
      assertEquals(results(kernel).map(_.operation), Vector(
        AtomicOperation.FetchMin, AtomicOperation.FetchMax, AtomicOperation.FetchAnd,
        AtomicOperation.FetchOr, AtomicOperation.FetchXor, AtomicOperation.CompareExchange))
      assertEquals(results(kernel).map(_.order), Vector(MemoryOrder.Relaxed, MemoryOrder.Acquire,
        MemoryOrder.Release, MemoryOrder.AcquireRelease, MemoryOrder.SequentiallyConsistent, MemoryOrder.AcquireRelease))
      assertEquals(results(kernel).map(_.failureOrder), Vector(None, None, None, None, None, Some(MemoryOrder.Acquire)))
      assertEquals(results(kernel).map(_.operands.size), Vector(1, 1, 1, 1, 1, 2))
      assert(results(kernel).forall(_.scope == AtomicScope.Device))
      assert(all(kernel.body).forall(_.span != SourceSpan.Unknown))
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      for marker <- Vector("__nv_atomic_fetch_min", "__nv_atomic_fetch_max", "__nv_atomic_fetch_and",
          "__nv_atomic_fetch_or", "__nv_atomic_fetch_xor", "__nv_atomic_compare_exchange_n",
          "__NV_THREAD_SCOPE_DEVICE", "__NV_ATOMIC_ACQ_REL", "__NV_ATOMIC_SEQ_CST") do
        assert(source.contains(marker), marker)
    }

  test("block scope admits shared and global integral targets for every operation"):
    accepted("""
      val tile = sharedArray[Int](2)
      val lane = threadIdx.x
      atomicBlock(tile(0)).store(p._3, MemoryOrder.Relaxed)
      val merged = atomicBlock(tile(0)).fetchOr(3, MemoryOrder.Relaxed)
      val observed = atomicBlock(tile(1)).compareExchange(0, merged, MemoryOrder.Relaxed, MemoryOrder.Relaxed)
      val low = atomicBlock(p._2(0)).fetchMin(observed, MemoryOrder.Relaxed)
      val high = atomicBlock(p._2(1)).fetchMax(lane, MemoryOrder.Relaxed)
      val masked = atomicBlock(tile(0)).fetchAnd(high, MemoryOrder.Relaxed)
      val flipped = atomicBlock(tile(1)).fetchXor(masked, MemoryOrder.Relaxed)
      if lane == 0 then p._2(2) = low + flipped
    """) { kernel =>
      assertEquals(results(kernel).map(_.operation), Vector(
        AtomicOperation.FetchOr, AtomicOperation.CompareExchange, AtomicOperation.FetchMin,
        AtomicOperation.FetchMax, AtomicOperation.FetchAnd, AtomicOperation.FetchXor))
      assert(results(kernel).forall(_.scope == AtomicScope.Block))
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      assert(source.contains("__NV_THREAD_SCOPE_BLOCK"))
    }

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"cells\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedScopedIntegralAtomics", factory(body, schema, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("compareExchange failure orders follow the memory-model legality matrix"):
    def cas(success: String, failure: String): String =
      s"val observed = atomicDevice(p._2(0)).compareExchange(0, 1, MemoryOrder.$success, MemoryOrder.$failure); p._2(1) = observed"
    rejected(cas("Relaxed", "Acquire"), "failure orders")
    rejected(cas("Release", "Acquire"), "failure orders")
    rejected(cas("SequentiallyConsistent", "Release"), "failure orders")
    rejected(cas("AcquireRelease", "AcquireRelease"), "failure orders")
    rejected(cas("AcquireRelease", "SequentiallyConsistent"), "failure orders")

  test("integral operations reject Float and Double elements"):
    rejected("val low = atomicDevice(p._2(0)).fetchMin(0.5f, MemoryOrder.Relaxed); p._2(1) = low",
      "Int elements only",
      parameters = "input[Float](\"data\"), inOut[Float](\"cells\"), value[Int](\"count\")")
    rejected("val observed = atomicDevice(p._2(0)).compareExchange(0.0, 1.0, MemoryOrder.Relaxed, MemoryOrder.Relaxed); p._2(1) = observed",
      "Int elements only",
      parameters = "input[Double](\"data\"), inOut[Double](\"cells\"), value[Int](\"count\")")

  test("success and failure orders must be direct MemoryOrder case references"):
    rejected("val low = atomicDevice(p._2(0)).fetchMin(1, Host.order); p._2(1) = low",
      "direct MemoryOrder case references",
      "object Host { def order: flight4s.core.ir.MemoryOrder = { println(1); flight4s.core.ir.MemoryOrder.Relaxed } }")
    rejected("""val observed = atomicDevice(p._2(0)).compareExchange(0, 1, MemoryOrder.Relaxed,
      if p._3 > 0 then MemoryOrder.Relaxed else MemoryOrder.Acquire); p._2(1) = observed""",
      "direct MemoryOrder case references")

  test("device scope still requires global targets and results stay initializer-only"):
    rejected("val tile = sharedArray[Int](1); val low = atomicDevice(tile(0)).fetchMin(1, MemoryOrder.Relaxed); p._2(0) = low",
      "global")
    rejected("p._2(0) = atomicDevice(p._2(1)).fetchOr(1, MemoryOrder.Relaxed)", "directly initialize")
    rejected("if atomicDevice(p._2(0)).compareExchange(0, 1, MemoryOrder.Relaxed, MemoryOrder.Relaxed) == 0 then p._2(1) = 1",
      "directly initialize")

  test("host calls fail on integral atomic markers"):
    val view = new ScalaKernel.DeviceScopedAtomic[Int]()
    intercept[IllegalStateException](view.fetchMin(1, MemoryOrder.Relaxed))
    intercept[IllegalStateException](view.fetchMax(1, MemoryOrder.Relaxed))
    intercept[IllegalStateException](view.fetchAnd(1, MemoryOrder.Relaxed))
    intercept[IllegalStateException](view.fetchOr(1, MemoryOrder.Relaxed))
    intercept[IllegalStateException](view.fetchXor(1, MemoryOrder.Relaxed))
    intercept[IllegalStateException](view.compareExchange(0, 1, MemoryOrder.Relaxed, MemoryOrder.Relaxed))
