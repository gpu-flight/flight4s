package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelAtomicCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"bins\"), value[Int](\"count\")"): String = s"""
    package quotedatomicfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("atomics", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"bins\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Atomics", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedatomicfixture.Definitions")
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

  test("atomic add stages one statement on a global element without a result local"):
    accepted("atomic(p._2(p._1(threadIdx.x) % 8)).add(1)") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicAdd[?, ?]]), 1)
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicFetchAdd[?, ?]]), 0)
      assert(all(kernel.body).forall(_.span != SourceSpan.Unknown))
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      assertEquals(source.sliding("atomicAdd(&".length).count(_ == "atomicAdd(&"), 1)
    }

  test("atomic fetchAdd binds the old value once and reuses it as an ordinary local"):
    accepted("val slot = atomic(p._2(0)).fetchAdd(1); if slot < p._3 then p._2(slot) = slot") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicFetchAdd[?, ?]]), 1)
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      assertEquals(source.sliding("atomicAdd(&".length).count(_ == "atomicAdd(&"), 1)
    }

  test("Float values and rank-1 shared elements stage atomics with uniform-quiet diagnostics"):
    accepted("""
      val tile = sharedArray[Float](1)
      val lane = threadIdx.x
      sync.blockAfter { if lane == 0 then tile(0) = 0.0f }
      val item = if lane < p._3 then p._1(lane) else 0.0f
      atomic(tile(0)).add(item)
      val before = atomic(tile(0)).fetchAdd(item)
      sync.block()
      if lane == 0 then p._2(0) = tile(0) + before
    """, parameters = "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicAdd[?, ?]]), 1)
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicFetchAdd[?, ?]]), 1)
      assertEquals(kernel.sharedMemory.size, 1)
    }

  test("atomics stay divergence-safe under lane-varying branches"):
    accepted("if threadIdx.x < p._3 then atomic(p._2(threadIdx.x % 4)).add(1)") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  test("qualified and renamed imported atomic factories share the same lowering"):
    accepted("""
      flight4s.frontend.ScalaKernel.atomic(p._2(0)).add(1)
      exchangeAt(p._2(1)).add(2)
    """, schema = "import flight4s.frontend.ScalaKernel.{atomic as exchangeAt}") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[AtomicAdd[?, ?]]), 2)
    }

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Int](\"data\"), inOut[Int](\"bins\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedAtomics", factory(body, schema, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("targets must be read-write buffer elements or rank-1 shared elements"):
    rejected("atomic(p._1(0)).add(1)", "read-write")
    rejected("var total = 0; atomic(total).add(1)", "buffer element or rank-1 shared")
    rejected("val item = p._1(0); atomic(item).add(1)", "buffer element or rank-1 shared")
    rejected("atomic(p._3).add(1)", "buffer element or rank-1 shared")

  test("element types stay Int and Float only"):
    rejected("atomic(p._2(0)).add(1.0)", "Int and Float",
      parameters = "input[Double](\"data\"), inOut[Double](\"bins\"), value[Int](\"count\")")
    rejected("atomic(p._2(0)).add(true)", "Int and Float",
      parameters = "input[Boolean](\"data\"), inOut[Boolean](\"bins\"), value[Int](\"count\")")

  test("atomic views are consumed immediately and never escape"):
    rejected("val handle = atomic(p._2(0)); handle.add(1)", "consumed immediately")
    rejected("val result = atomic(p._2(0)).add(1)", "primitive locals")
    rejected("p._2(0) = atomic(p._2(0)).fetchAdd(1)", "directly initialize")
    rejected("val next = atomic(p._2(0)).fetchAdd(1) + 1; p._2(0) = next", "directly initialize")
    rejected("val unused = deviceRange(0, 0).foldLeft(0)((sum, i) => sum + atomic(p._2(0)).fetchAdd(1))", "directly initialize")

  test("host lookalikes and effectful receivers are rejected"):
    rejected("val slot = Host.atomic(p._2(0)).fetchAdd(1); p._2(0) = slot", "host/helper calls",
      "object Host { class Ref(value: Int) { def fetchAdd(delta: Int): Int = value }; def atomic(value: Int): Ref = { println(1); new Ref(value) } }")
    rejected("Host.api.atomic(p._2(0)).add(1)", "declared where they are used",
      "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("host calls fail on atomic markers"):
    intercept[IllegalStateException](ScalaKernel.atomic(1))
    val view = new ScalaKernel.DeviceAtomic[Int]()
    intercept[IllegalStateException](view.add(1))
    intercept[IllegalStateException](view.fetchAdd(1))
