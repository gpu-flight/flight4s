package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*
import flight4s.core.types.{U32, UInt}

class ScalaKernelWarpShuffleCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): String = s"""
    package quotedwarpshufflefixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("shuffle", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "WarpShuffle", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedwarpshufflefixture.Definitions")
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

  private def shuffles(kernel: Kernel[?]): Vector[WarpShuffle[?, ?]] =
    all(kernel.body).collect { case shuffle: WarpShuffle[?, ?] => shuffle }

  private def maskBits(shuffle: WarpShuffle[?, ?]): Int = shuffle.mask match
    case Literal(mask: UInt, U32, _) => mask.toIntBits
    case other => fail(s"not a constant shuffle mask: $other")

  test("all four variants stage warp shuffles with their operators masks and widths"):
    accepted("""
      val item = p._1(threadIdx.x)
      val first = warpShuffle(-1, item, 0)
      val above = warpShuffleUp(-1, item, 1)
      val below = warpShuffleDown(0x0000ffff, item, 2, 16)
      val partner = warpShuffleXor(sync.fullMask, item, 3, 8)
      p._2(threadIdx.x) = first + above + below + partner
    """) { kernel =>
      assertEquals(shuffles(kernel).map(_.operator.cudaName), Vector(
        "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync"))
      assertEquals(shuffles(kernel).map(_.width), Vector(32, 32, 16, 8))
      assertEquals(shuffles(kernel).map(maskBits), Vector(-1, -1, 0xffff, -1))
      assert(shuffles(kernel).forall(_.span != SourceSpan.Unknown))
      val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
      for marker <- Vector("::__shfl_sync(", "::__shfl_up_sync(", "::__shfl_down_sync(", "::__shfl_xor_sync(") do
        assertEquals(source.sliding(marker.length).count(_ == marker), 1, marker)
      assert(!source.contains("__syncthreads"))
      assert(!source.contains("__syncwarp"))
    }

  test("direct source lanes admit computed Int expressions and literal broadcasts"):
    accepted("""
      val lane = threadIdx.x
      val item = p._1(lane)
      val source = (lane + 1) % 32
      val next = warpShuffle(-1, item, source)
      val first = warpShuffle(-1, item, 0)
      p._2(lane) = next + first
    """) { kernel =>
      assertEquals(shuffles(kernel).size, 2)
      assert(shuffles(kernel).head.selector match { case _: Literal[?] => false; case _ => true })
    }

  test("Int and Double values shuffle with vars loops and renamed imports"):
    accepted("""
      var total = 0.0
      for round <- deviceRange(0, p._3) do
        val next = fetch(-1, total, 1)
        total += next
      val direct = flight4s.frontend.ScalaKernel.warpShuffleXor(-1, total, 1)
      p._2(threadIdx.x) = direct
    """, schema = "import flight4s.frontend.ScalaKernel.{warpShuffleUp as fetch}",
      parameters = "input[Double](\"data\"), output[Double](\"target\"), value[Int](\"count\")") { kernel =>
      assertEquals(shuffles(kernel).map(_.operator.cudaName), Vector("__shfl_up_sync", "__shfl_xor_sync"))
    }

  test("divergent control warns per shuffle while block-uniform control stays quiet"):
    accepted("if threadIdx.x < 16 then { val next = warpShuffle(0x0000ffff, p._1(0), 0, 16); p._2(0) = next }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings.map(_.code),
        Vector(ValidationWarningCode.WarpParticipationMayDiverge))
    }
    accepted("if blockIdx.x == 0 then { val next = warpShuffle(-1, p._1(0), 0); p._2(0) = next }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedWarpShuffle", factory(body, schema, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("masks widths and selectors keep their compile-time bounds"):
    rejected("val next = warpShuffle(0, p._1(0), 0); p._2(0) = next", "at least one participating lane")
    rejected("val next = warpShuffle(p._3, p._1(0), 0); p._2(0) = next", "compile-time constant Int")
    rejected("val next = warpShuffle(-1, p._1(0), 0, 3); p._2(0) = next", "widths must be compile-time constants")
    rejected("val next = warpShuffle(-1, p._1(0), 0, p._3); p._2(0) = next", "widths must be compile-time constants")
    rejected("val next = warpShuffle(-1, p._1(0), -1); p._2(0) = next", "nonnegative")
    rejected("val next = warpShuffleUp(-1, p._1(0), p._3); p._2(0) = next", "compile-time Int constants in 0 to 31")
    rejected("val next = warpShuffleUp(-1, p._1(0), 32); p._2(0) = next", "compile-time Int constants in 0 to 31")
    rejected("val next = warpShuffleDown(-1, p._1(0), -1); p._2(0) = next", "compile-time Int constants in 0 to 31")
    rejected("val next = warpShuffleXor(-1, p._1(0), 32); p._2(0) = next", "compile-time Int constants in 0 to 31")
    rejected("val next = warpShuffleXor(-1, p._1(0), p._3); p._2(0) = next", "compile-time Int constants in 0 to 31")

  test("values stay matching primitives and shuffles cannot hide hosts or lookalikes"):
    rejected("val next = warpShuffle(-1, p._4, 0); p._2(0) = if next then 1.0f else 0.0f", "Int, Float and Double",
      parameters = "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\"), value[Boolean](\"flag\")")
    rejected("val next = warpShuffle(-1, Host.value, 0); p._2(0) = next", "captures", "object Host { val value = 1.0f }")
    rejected("val next = Host.warpShuffle(-1, p._1(0), 0); p._2(0) = next", "host/helper calls",
      "object Host { def warpShuffle(mask: Int, value: Float, lane: Int): Float = { println(1); value } }")
    rejected("val next = Host.api.warpShuffleUp(-1, p._1(0), 1); p._2(0) = next", "host/helper calls",
      "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("shuffles are initializer-only and cannot embed in expressions or callbacks"):
    rejected("p._2(0) = warpShuffle(-1, p._1(0), 0)", "directly initialize")
    rejected("val next = warpShuffleXor(-1, p._1(0), 1) + 1.0f; p._2(0) = next", "directly initialize")
    rejected("if warpShuffleDown(-1, p._1(0), 1) > 0.0f then p._2(0) = 1.0f", "directly initialize")
    rejected("val unused = deviceRange(0, 0).foldLeft(0.0f)((sum, i) => sum + warpShuffleUp(-1, p._1(0), 1))", "directly initialize")

  test("host calls fail on warp shuffle markers"):
    intercept[IllegalStateException](ScalaKernel.warpShuffle(-1, 1.0f, 0))
    intercept[IllegalStateException](ScalaKernel.warpShuffleUp(-1, 1, 1))
    intercept[IllegalStateException](ScalaKernel.warpShuffleDown(-1, 1.0, 1, 8))
    intercept[IllegalStateException](ScalaKernel.warpShuffleXor(-1, 1, 1))
