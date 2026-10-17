package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*
import flight4s.core.types.{U32, UInt}

class ScalaKernelWarpCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = ""): String = s"""
    package quotedwarpfixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("warp", params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "Warp", factory(body, schema))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedwarpfixture.Definitions")
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

  private def maskBits(statement: Stmt): Int = statement match
    case WarpBarrier(Literal(mask: UInt, U32, span), barrierSpan) =>
      assertNotEquals(span, SourceSpan.Unknown)
      assertNotEquals(barrierSpan, SourceSpan.Unknown)
      mask.toIntBits
    case other => fail(s"not a constant-mask warp barrier: $other")

  private def syncwarpCount(kernel: Kernel[?]): Int =
    val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
    assert(!source.contains("__syncthreads"))
    source.sliding("::__syncwarp(".length).count(_ == "::__syncwarp(")

  test("standalone warp sync lowers literal inline-val and hexadecimal masks to warp barriers"):
    accepted("sync.warp(-1); sync.warp(0xffffffff); sync.warp(sync.fullMask); sync.warp(0x0000ffff)") { kernel =>
      assertEquals(kernel.body.statements.map(maskBits), Vector(-1, -1, -1, 0xffff))
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      assertEquals(source.sliding("::__syncwarp(0xffffffffu);".length).count(_ == "::__syncwarp(0xffffffffu);"), 3)
      assertEquals(source.sliding("::__syncwarp(0x0000ffffu);".length).count(_ == "::__syncwarp(0x0000ffffu);"), 1)
      assertEquals(syncwarpCount(kernel), 4)
    }

  test("warpAfter emits its body once in a scope followed by exactly one trailing warp barrier"):
    accepted("p._2(0) = 1; sync.warpAfter(-1) { p._2(1) = 2 }; p._2(2) = 3") { kernel =>
      kernel.body.statements match
        case Vector(_: Store[?, ?], scope: ScopedBlock, barrier: WarpBarrier, _: Store[?, ?]) =>
          assertEquals(scope.body.statements.size, 1)
          assert(scope.body.statements.head.isInstanceOf[Store[?, ?]])
          assertEquals(barrier.span, scope.span)
          assertNotEquals(barrier.span, SourceSpan.Unknown)
          assertEquals(maskBits(barrier), -1)
        case other => fail(s"unexpected warp phase lowering: $other")
      assertEquals(syncwarpCount(kernel), 1)
    }

  test("empty nested and explicitly synchronized warp phases retain every trailing barrier"):
    accepted("sync.warpAfter(-1) { () }; sync.warpAfter(-1) { sync.warpAfter(0x0000ffff) { () }; sync.warp(-1) }") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[ScopedBlock]), 3)
      assertEquals(all(kernel.body).count(_.isInstanceOf[WarpBarrier]), 4)
      assertEquals(kernel.body.statements.map(_.getClass.getSimpleName), Vector("ScopedBlock", "WarpBarrier", "ScopedBlock", "WarpBarrier"))
      assertEquals(syncwarpCount(kernel), 4)
    }

  test("warp and block synchronization mix without deduplication or reordering"):
    accepted("sync.warpAfter(-1) { sync.block() }; sync.blockAfter { sync.warp(-1) }") { kernel =>
      assertEquals(kernel.body.statements.map(_.getClass.getSimpleName), Vector("ScopedBlock", "WarpBarrier", "ScopedBlock", "Barrier"))
      val first = kernel.body.statements(0).asInstanceOf[ScopedBlock]
      assertEquals(first.body.statements.map(_.getClass.getSimpleName), Vector("Barrier"))
      val second = kernel.body.statements(2).asInstanceOf[ScopedBlock]
      assertEquals(second.body.statements.map(_.getClass.getSimpleName), Vector("WarpBarrier"))
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      assertEquals(source.sliding("__syncthreads();".length).count(_ == "__syncthreads();"), 2)
      assertEquals(source.sliding("::__syncwarp(".length).count(_ == "::__syncwarp("), 2)
    }

  test("qualified and renamed imported warp symbols share the same lowering"):
    accepted("flight4s.frontend.ScalaKernel.sync.warpAfter(-1) { () }; warpAfter(-1) { () }; publish(-1) { () }; coordination.warpAfter(-1) { () }; warp(fullMask); coordination.warp(-1)",
      "import flight4s.frontend.ScalaKernel.sync.*; import flight4s.frontend.ScalaKernel.sync.{warpAfter as publish}; import flight4s.frontend.ScalaKernel.{sync as coordination}") { kernel =>
      assertEquals(all(kernel.body).count(_.isInstanceOf[WarpBarrier]), 6)
      assertEquals(all(kernel.body).collect { case barrier: WarpBarrier => maskBits(barrier) }, Vector.fill(6)(-1))
    }

  test("unconditional warp phases can declare shared storage with lexical handle visibility"):
    accepted("sync.warpAfter(-1) { val tile = sharedArray[Int](32); tile(threadIdx.x) = 1; sync.warpAfter(-1) { p._2(threadIdx.x) = tile(threadIdx.x) } }") { kernel =>
      assertEquals(kernel.sharedMemory.size, 1)
      assertEquals(all(kernel.body).count(_.isInstanceOf[WarpBarrier]), 2)
    }

  test("lane-varying statements inside a warp phase do not guard its trailing barrier"):
    accepted("sync.warpAfter(-1) { if threadIdx.x < p._3 then p._2(threadIdx.x) = 1 }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
      val scope = kernel.body.statements.head.asInstanceOf[ScopedBlock]
      assert(scope.body.statements.head.isInstanceOf[IfThen])
      assert(kernel.body.statements(1).isInstanceOf[WarpBarrier])
    }

  test("divergent enclosing control warns at the warp phase call including nested phases"):
    accepted("val lane = threadIdx.x; if lane < 32 then sync.warpAfter(-1) { sync.warpAfter(-1) { () } }") { kernel =>
      val barriers = all(kernel.body).collect { case barrier: WarpBarrier => barrier }
      val warnings = KernelValidator.validate(kernel).warnings
      assertEquals(warnings.map(_.code), Vector.fill(2)(ValidationWarningCode.WarpParticipationMayDiverge))
      assertEquals(warnings.map(_.span), barriers.map(_.span))
      assert(warnings.forall(_.span != SourceSpan.Unknown))
    }
    accepted("deviceRange(0, threadIdx.x).foreach { i => sync.warpAfter(-1) { () } }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings.map(_.code), Vector(ValidationWarningCode.WarpParticipationMayDiverge))
    }

  test("block-uniform branches and serial loops preserve the warp participation warning policy"):
    accepted("""
      if blockIdx.x == 0 then sync.warpAfter(-1) { () }
      for i <- 0 until p._3 do sync.warpAfter(-1) { () }
    """) { kernel => assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty) }

  private def rejected(body: String, message: String, schema: String = ""): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedWarp", factory(body, schema))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("empty dynamic computed and captured masks are rejected as compile-time constants"):
    rejected("sync.warp(0)", "at least one participating lane")
    rejected("sync.warpAfter(0) { () }", "at least one participating lane")
    rejected("sync.warp(p._3)", "compile-time constant Int")
    rejected("sync.warpAfter(p._3) { () }", "compile-time constant Int")
    rejected("val mask = -1; sync.warp(mask)", "compile-time constant Int")
    rejected("sync.warp(1 << 16)", "compile-time constant Int")
    rejected("sync.warp(Host.mask)", "compile-time constant Int", "object Host { val mask = -1 }")

  test("warp phases cannot hide host effects helpers captures lookalikes or effectful receivers"):
    rejected("sync.warpAfter(-1) { Host.work() }", "host effects", "object Host { def work(): Unit = println(1) }")
    rejected("Host.warp(-1)", "host effects", "object Host { def warp(mask: Int): Unit = println(1) }")
    rejected("Host.warpAfter(-1) { p._2(0) = 1 }", "host effects", "object Host { def warpAfter(mask: Int)(body: => Unit): Unit = body }")
    rejected("Host.api.warpAfter(-1) { () }", "host effects", "object Host { def api: flight4s.frontend.ScalaKernel.sync.type = { println(1); flight4s.frontend.ScalaKernel.sync } }")
    rejected("Host.api.sync.warp(-1)", "host effects", "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("conditional or repeated warp phases cannot bypass shared-declaration restrictions"):
    for body <- Vector(
        "if p._3 > 0 then sync.warpAfter(-1) { val tile = sharedArray[Int](32); () }",
        "for i <- 0 until 2 do sync.warpAfter(-1) { val tile = sharedArray[Int](32); () }",
        "deviceRange(0, 2).foreach { i => sync.warpAfter(-1) { val tile = sharedArray[Int](32); () } }") do
      rejected(body, "outside branches and loops")

  test("warp synchronization is statement-only and cannot leak locals or hide effects in pure callbacks"):
    rejected("sync.warpAfter(-1) { val inner = 1 }; p._2(0) = inner", "Not found: inner")
    rejected("val result = sync.warp(-1)", "primitive locals")
    rejected("val result = sync.warpAfter(-1) { p._2(0) = 1 }", "primitive locals")
    rejected("val unused = deviceRange(0, 0).map(i => { sync.warp(-1); i })", "expression blocks")
    rejected("val unused = deviceRange(0, 0).foldLeft(0)((sum, i) => { sync.warpAfter(-1) { () }; sum })", "expression blocks")

  test("host calls fail without evaluating the by-name warp body"):
    var executed = false
    intercept[IllegalStateException](ScalaKernel.sync.warpAfter(-1) { executed = true })
    assert(!executed)
    intercept[IllegalStateException](ScalaKernel.sync.warp(-1))
    assertEquals(ScalaKernel.sync.fullMask, -1)
