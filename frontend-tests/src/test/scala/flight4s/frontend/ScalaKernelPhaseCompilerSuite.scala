package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ScalaKernelPhaseCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = ""): String = s"""
    package quotedphasefixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("phase", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Phase", factory(body, schema))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedphasefixture.Definitions")
        val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance()).asInstanceOf[Kernel[Tuple]]
        assert(KernelValidator.validate(staged).isValid)
        check(staged)
      }
    }

  test("phase emits its body once in a scope followed by exactly one barrier"):
    accepted("p._2(0) = 1; block.phase { p._2(1) = 2 }; p._2(2) = 3") { kernel =>
      kernel.body.statements match
        case Vector(_: Store[?, ?], scope: ScopedBlock, barrier: Barrier, _: Store[?, ?]) =>
          assertEquals(scope.body.statements.size, 1)
          assert(scope.body.statements.head.isInstanceOf[Store[?, ?]])
          assertEquals(barrier.span, scope.span)
          assertNotEquals(barrier.span, SourceSpan.Unknown)
        case other => fail(s"unexpected phase lowering: $other")
      val generated = CudaCodegen.generate(kernel).toOption.get
      assertEquals(generated.cudaSource.sliding("__syncthreads();".length).count(_ == "__syncthreads();"), 1)
    }

  private def all(body: Block): Vector[Stmt] = body.statements.flatMap {
    case scope: ScopedBlock => Vector(scope) ++ all(scope.body)
    case branch: IfThen => Vector(branch) ++ all(branch.thenBlock) ++ branch.elseBlock.toVector.flatMap(all)
    case loop: ForLoop => Vector(loop) ++ all(loop.body)
    case statement => Vector(statement)
  }

  test("empty nested and explicitly synchronized phases retain every trailing barrier"):
    accepted("block.phase { () }; block.phase { block.phase { () }; barrier() }") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[ScopedBlock]), 3)
      assertEquals(all(kernel.body).count(_.isInstanceOf[Barrier]), 4)
      assertEquals(kernel.body.statements.map(_.getClass.getSimpleName), Vector("ScopedBlock", "Barrier", "ScopedBlock", "Barrier"))
      val generated = CudaCodegen.generate(kernel).toOption.get
      assertEquals(generated.cudaSource.sliding("__syncthreads();".length).count(_ == "__syncthreads();"), 4)
    }

  test("phase locals shadow independently while enclosing vars and shared aliases remain usable"):
    accepted("""
      val tile = sharedArray[Int](64)
      var total = 1
      block.phase {
        val saved = total;
        {
          val total = saved + 1
          val alias = tile
          alias(threadIdx.x) = total
        }
      }
      block.phase { total += tile(threadIdx.x) }
      p._2(threadIdx.x) = total
    """) { kernel =>
      val locals = all(kernel.body).collect { case declaration: LocalDeclaration[?] => declaration.local.name }
      assertEquals(locals.distinct.size, 3)
      assertEquals(kernel.sharedMemory.size, 1)
    }

  test("unconditional phases can declare shared storage with lexical handle visibility"):
    accepted("block.phase { val tile = sharedArray[Int](64); tile(threadIdx.x) = 1; block.phase { p._2(threadIdx.x) = tile(threadIdx.x) } }") { kernel =>
      assertEquals(kernel.sharedMemory.size, 1)
      assertEquals(all(kernel.body).count(_.isInstanceOf[Barrier]), 2)
    }

  test("qualified and renamed imported phase symbols share the same lowering"):
    accepted("flight4s.frontend.ScalaKernel.block.phase { () }; sync { () }; threads.phase { () }",
      "import flight4s.frontend.ScalaKernel.block.{phase as sync}; import flight4s.frontend.ScalaKernel.{block as threads}") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[Barrier]), 3)
    }

  test("lane-varying statements inside a phase do not guard its trailing barrier"):
    accepted("block.phase { if threadIdx.x < p._3 then p._2(threadIdx.x) = 1 }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
      val scope = kernel.body.statements.head.asInstanceOf[ScopedBlock]
      assert(scope.body.statements.head.isInstanceOf[IfThen])
      assert(kernel.body.statements(1).isInstanceOf[Barrier])
    }

  test("divergent enclosing control warns at the phase call including nested phases"):
    accepted("val lane = threadIdx.x; if lane < 32 then block.phase { block.phase { () } }") { kernel =>
      val barriers = all(kernel.body).collect { case barrier: Barrier => barrier }
      val warnings = KernelValidator.validate(kernel).warnings
      assertEquals(warnings.map(_.code), Vector.fill(2)(ValidationWarningCode.BarrierMayDiverge))
      assertEquals(warnings.map(_.span), barriers.map(_.span))
      assert(warnings.forall(_.span != SourceSpan.Unknown))
    }
    accepted("deviceRange(0, threadIdx.x).foreach { i => block.phase { () } }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))
    }

  test("block-uniform branches serial loops and guarded traversal phases preserve warning policy"):
    accepted("""
      if blockIdx.x == 0 then block.phase { () }
      for i <- 0 until p._3 do block.phase { () }
      for i <- deviceRange(0, p._3) if i % 2 == 0 do block.phase { () }
    """) { kernel => assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty) }

  private def rejected(body: String, message: String, schema: String = ""): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedPhase", factory(body, schema))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("phases cannot hide host effects helpers captures lookalikes or effectful receivers"):
    rejected("block.phase { Host.work() }", "host effects", "object Host { def work(): Unit = println(1) }")
    rejected("block.phase { p._2(0) = Host.value }", "captures", "object Host { val value = 1 }")
    rejected("Host.phase { p._2(0) = 1 }", "host effects", "object Host { def phase(body: => Unit): Unit = body }")
    rejected("Host.api.phase { () }", "host effects", "object Host { def api: flight4s.frontend.ScalaKernel.block.type = { println(1); flight4s.frontend.ScalaKernel.block } }")
    rejected("Host.api.block.phase { () }", "host effects", "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("conditional or repeated phases cannot bypass shared-declaration restrictions"):
    for body <- Vector(
        "if p._3 > 0 then block.phase { val tile = sharedArray[Int](64); () }",
        "for i <- 0 until 2 do block.phase { val tile = sharedArray[Int](64); () }",
        "deviceRange(0, 2).foreach { i => block.phase { val tile = sharedArray[Int](64); () } }") do
      rejected(body, "outside branches and loops")

  test("phase is statement-only and cannot leak locals or hide effects in pure callbacks"):
    rejected("block.phase { val inner = 1 }; p._2(0) = inner", "Not found: inner")
    rejected("val result = block.phase { p._2(0) = 1 }", "primitive locals")
    rejected("val unused = deviceRange(0, 0).map(i => { block.phase { p._2(0) = i }; i })", "expression blocks")
    rejected("val unused = deviceRange(0, 0).foldLeft(0)((sum, i) => { block.phase { () }; sum })", "expression blocks")
    rejected("block.phase { p._1(0) = 1 }", "Cannot prove")

  test("host calls fail without evaluating the by-name phase body"):
    var executed = false
    intercept[IllegalStateException](ScalaKernel.block.phase { executed = true })
    assert(!executed)
