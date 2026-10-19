package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelScopedAtomicCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"cells\"), value[Int](\"count\")"): String = s"""
    package quotedscopedatomicfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    import flight4s.core.ir.MemoryOrder
    $schema
    class Definitions:
      @experimental
      def definition = kernel("scoped", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"cells\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "ScopedAtomics", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedscopedatomicfixture.Definitions")
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

  test("all five scoped operations stage with explicit device scope and orders"):
    accepted("""
      atomicDevice(p._2(0)).store(1, MemoryOrder.Release)
      val seen = atomicDevice(p._2(0)).load(MemoryOrder.Acquire)
      val previous = atomicDevice(p._2(1)).exchange(seen, MemoryOrder.Relaxed)
      val before = atomicDevice(p._2(2)).fetchAdd(previous, MemoryOrder.SequentiallyConsistent)
      val after = atomicDevice(p._2(3)).fetchSub(before, MemoryOrder.Relaxed)
      p._2(4) = after
    """) { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicStore[?, ?]]), 1)
      assertEquals(results(kernel).map(_.operation), Vector(
        AtomicOperation.Load, AtomicOperation.Exchange, AtomicOperation.FetchAdd, AtomicOperation.FetchSub))
      assertEquals(results(kernel).map(_.order), Vector(MemoryOrder.Acquire, MemoryOrder.Relaxed,
        MemoryOrder.SequentiallyConsistent, MemoryOrder.Relaxed))
      assert(results(kernel).forall(_.scope == AtomicScope.Device))
      assert(all(kernel.body).forall(_.span != SourceSpan.Unknown))
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      for marker <- Vector("__nv_atomic_store", "__nv_atomic_load", "__nv_atomic_exchange",
          "__nv_atomic_fetch_add", "__nv_atomic_fetch_sub", "__NV_THREAD_SCOPE_DEVICE",
          "__NV_ATOMIC_RELEASE", "__NV_ATOMIC_ACQUIRE", "__NV_ATOMIC_SEQ_CST") do
        assert(source.contains(marker), marker)
    }

  test("block scope admits shared and global targets with Float and Double elements"):
    accepted("""
      val tile = sharedArray[Double](2)
      val lane = threadIdx.x
      atomicBlock(tile(0)).store(0.5, MemoryOrder.Relaxed)
      val old = atomicBlock(tile(1)).fetchAdd(0.25, MemoryOrder.Relaxed)
      val global = atomicBlock(p._2(0)).fetchSub(old, MemoryOrder.Relaxed)
      if lane == 0 then p._2(1) = global
    """, parameters = "input[Double](\"data\"), inOut[Double](\"cells\"), value[Int](\"count\")") { kernel =>
      assert(results(kernel).forall(_.scope == AtomicScope.Block))
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicStore[?, ?]]), 1)
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      assert(source.contains("__NV_THREAD_SCOPE_BLOCK"))
    }

  test("bare imported orders and renamed factories share the same lowering"):
    accepted("""
      publish(p._2(0)).store(1, Release)
      val seen = flight4s.frontend.ScalaKernel.atomicDevice(p._2(0)).load(Acquire)
      p._2(1) = seen
    """, schema = "import flight4s.core.ir.MemoryOrder.*; import flight4s.frontend.ScalaKernel.{atomicDevice as publish}") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicStore[?, ?]]), 1)
      assertEquals(results(kernel).map(_.operation), Vector(AtomicOperation.Load))
    }

  test("scoped atomics stay divergence-safe under lane-varying branches"):
    accepted("if threadIdx.x < p._3 then atomicDevice(p._2(0)).store(1, MemoryOrder.Relaxed)") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"cells\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedScopedAtomics", factory(body, schema, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("stores and loads reject the orders the memory model forbids"):
    rejected("atomicDevice(p._2(0)).store(1, MemoryOrder.Acquire)", "stores allow")
    rejected("atomicDevice(p._2(0)).store(1, MemoryOrder.AcquireRelease)", "stores allow")
    rejected("val seen = atomicDevice(p._2(0)).load(MemoryOrder.Release); p._2(0) = seen", "loads allow")
    rejected("val seen = atomicDevice(p._2(0)).load(MemoryOrder.AcquireRelease); p._2(0) = seen", "loads allow")

  test("orders must be direct MemoryOrder case references"):
    rejected("atomicDevice(p._2(0)).store(1, if p._3 > 0 then MemoryOrder.Relaxed else MemoryOrder.Release)", "direct MemoryOrder case references")
    rejected("atomicDevice(p._2(0)).store(1, Host.order)", "direct MemoryOrder case references",
      "object Host { def order: flight4s.core.ir.MemoryOrder = { println(1); flight4s.core.ir.MemoryOrder.Relaxed } }")

  test("scopes constrain targets and elements stay atomic value types"):
    rejected("val tile = sharedArray[Int](1); atomicDevice(tile(0)).store(1, MemoryOrder.Relaxed)", "global")
    rejected("atomicDevice(p._1(0)).store(1, MemoryOrder.Relaxed)", "read-write")
    rejected("var x = 0; atomicBlock(x).store(1, MemoryOrder.Relaxed)", "buffer element or rank-1 shared")
    rejected("atomicDevice(p._2(0)).store(true, MemoryOrder.Relaxed)", "Int, Float and Double",
      parameters = "input[Boolean](\"data\"), inOut[Boolean](\"cells\"), value[Int](\"count\")")

  test("scoped views are consumed immediately and results stay initializer-only"):
    rejected("val view = atomicDevice(p._2(0)); view.store(1, MemoryOrder.Relaxed)", "consumed immediately")
    rejected("val result = atomicDevice(p._2(0)).store(1, MemoryOrder.Relaxed)", "primitive locals")
    rejected("p._2(0) = atomicDevice(p._2(1)).fetchAdd(1, MemoryOrder.Relaxed)", "directly initialize")
    rejected("if atomicDevice(p._2(0)).load(MemoryOrder.Acquire) == 1 then p._2(1) = 1", "directly initialize")

  test("host calls fail on scoped atomic markers"):
    intercept[IllegalStateException](ScalaKernel.atomicDevice(1))
    intercept[IllegalStateException](ScalaKernel.atomicBlock(1))
    val view = new ScalaKernel.DeviceScopedAtomic[Int]()
    intercept[IllegalStateException](view.store(1, MemoryOrder.Relaxed))
    intercept[IllegalStateException](view.load(MemoryOrder.Acquire))
    intercept[IllegalStateException](view.exchange(1, MemoryOrder.Relaxed))
    intercept[IllegalStateException](view.fetchAdd(1, MemoryOrder.Relaxed))
    intercept[IllegalStateException](view.fetchSub(1, MemoryOrder.Relaxed))
