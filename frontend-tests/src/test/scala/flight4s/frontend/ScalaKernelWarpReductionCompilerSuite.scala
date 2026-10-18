package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*
import flight4s.core.types.{U32, UInt}

class ScalaKernelWarpReductionCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): String = s"""
    package quotedwarpreducefixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("reduce", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "WarpReduce", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedwarpreducefixture.Definitions")
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

  private def shuffleCount(kernel: Kernel[?]): Int =
    val source = CudaCodegen.generate(kernel).toOption.get.cudaSource
    assert(!source.contains("__syncthreads"))
    assert(!source.contains("__syncwarp"))
    source.sliding("::__shfl_xor_sync".length).count(_ == "::__shfl_xor_sync")

  test("full-width Float sums stage one shuffle per tree level with the call-site span"):
    accepted("val total = warpReduceSum(-1, p._1(threadIdx.x)); p._2(threadIdx.x) = total") { kernel =>
      assertEquals(shuffles(kernel).size, 5)
      assertEquals(shuffles(kernel).map(maskBits), Vector.fill(5)(-1))
      assertEquals(shuffles(kernel).map(_.width), Vector.fill(5)(32))
      assert(shuffles(kernel).forall(_.span != SourceSpan.Unknown))
      assert(all(kernel.body).forall(_.span != SourceSpan.Unknown))
      assertEquals(shuffleCount(kernel), 5)
    }

  test("explicit power-of-two widths control tree depth including the identity width"):
    accepted("""
      val a = warpReduceSum(-1, p._1(threadIdx.x), 16)
      val b = warpReduceSum(0x0000ffff, p._1(threadIdx.x), 16)
      val c = warpReduceSum(-1, p._1(threadIdx.x), 2)
      val d = warpReduceSum(-1, p._1(threadIdx.x), 1)
      p._2(threadIdx.x) = a + b + c + d
    """) { kernel =>
      assertEquals(shuffles(kernel).map(_.width), Vector.fill(8)(16) :+ 2)
      assertEquals(shuffleCount(kernel), 9)
    }

  test("Int and Double values reduce with vars loops and the library fullMask constant"):
    accepted("""
      var total = 0.0
      for round <- deviceRange(0, p._3) do
        val sum = warpReduceSum(sync.fullMask, p._1(threadIdx.x), 8)
        total += sum
      p._2(threadIdx.x) = total
    """, parameters = "input[Double](\"data\"), output[Double](\"target\"), value[Int](\"count\")") { kernel =>
      assertEquals(shuffles(kernel).map(_.width), Vector.fill(3)(8))
      assertEquals(shuffles(kernel).map(maskBits), Vector.fill(3)(-1))
    }
    accepted("val total = warpReduceSum(0xff00ff00, p._1(threadIdx.x), 8); p._2(threadIdx.x) = total",
      parameters = "input[Int](\"data\"), output[Int](\"target\"), value[Int](\"count\")") { kernel =>
      assertEquals(shuffles(kernel).map(maskBits), Vector.fill(3)(0xff00ff00))
    }

  test("qualified and renamed imported reduction symbols share the same lowering"):
    accepted("""
      val a = flight4s.frontend.ScalaKernel.warpReduceSum(-1, p._1(0))
      val b = combine(-1, p._1(1))
      p._2(0) = a + b
    """, schema = "import flight4s.frontend.ScalaKernel.{warpReduceSum as combine}") { kernel =>
      assertEquals(shuffles(kernel).size, 10)
    }

  test("divergent enclosing control warns per shuffle while uniform control stays quiet"):
    accepted("if threadIdx.x < 16 then { val total = warpReduceSum(0x0000ffff, p._1(threadIdx.x), 2); p._2(threadIdx.x) = total }") { kernel =>
      val warnings = KernelValidator.validate(kernel).warnings
      assertEquals(warnings.map(_.code), Vector(ValidationWarningCode.WarpParticipationMayDiverge))
    }
    accepted("if blockIdx.x == 0 then { val total = warpReduceSum(-1, p._1(threadIdx.x), 2); p._2(threadIdx.x) = total }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedWarpReduce", factory(body, schema, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("masks and widths must form nonempty complete aligned compile-time groups"):
    rejected("val total = warpReduceSum(0, p._1(0)); p._2(0) = total", "at least one participating lane")
    rejected("val total = warpReduceSum(p._3, p._1(0)); p._2(0) = total", "compile-time constant Int")
    rejected("val mask = -1; val total = warpReduceSum(mask, p._1(0)); p._2(0) = total", "compile-time constant Int")
    rejected("val total = warpReduceSum(-1, p._1(0), 0); p._2(0) = total", "width")
    rejected("val total = warpReduceSum(-1, p._1(0), 3); p._2(0) = total", "width")
    rejected("val total = warpReduceSum(-1, p._1(0), 64); p._2(0) = total", "width")
    rejected("val total = warpReduceSum(-1, p._1(0), p._3); p._2(0) = total", "width")
    rejected("val total = warpReduceSum(0x0000ffff, p._1(0)); p._2(0) = total", "complete aligned groups")
    rejected("val total = warpReduceSum(0x55555555, p._1(0), 2); p._2(0) = total", "complete aligned groups")
    rejected("val total = warpReduceSum(0xff0, p._1(0), 8); p._2(0) = total", "complete aligned groups")

  test("values stay matching primitives and reductions cannot hide hosts or lookalikes"):
    rejected("val total = warpReduceSum(-1, p._4); p._2(0) = if total then 1.0f else 0.0f", "Int, Float and Double",
      parameters = "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\"), value[Boolean](\"flag\")")
    rejected("val total = warpReduceSum(-1, Host.value); p._2(0) = total", "captures", "object Host { val value = 1.0f }")
    rejected("val total = Host.warpReduceSum(-1, p._1(0)); p._2(0) = total", "host/helper calls",
      "object Host { def warpReduceSum(mask: Int, value: Float): Float = { println(1); value } }")
    rejected("val total = Host.api.warpReduceSum(-1, p._1(0)); p._2(0) = total", "host/helper calls",
      "object Host { def api: flight4s.frontend.ScalaKernel.type = { println(1); flight4s.frontend.ScalaKernel } }")

  test("reductions are initializer-only and cannot embed in expressions or callbacks"):
    rejected("p._2(0) = warpReduceSum(-1, p._1(0))", "directly initialize")
    rejected("val total = warpReduceSum(-1, p._1(0)) + 1.0f; p._2(0) = total", "directly initialize")
    rejected("if warpReduceSum(-1, p._1(0)) > 0.0f then p._2(0) = 1.0f", "directly initialize")
    rejected("val unused = deviceRange(0, 0).map(i => { val total = warpReduceSum(-1, p._1(0)); i })", "expression blocks")
    rejected("val unused = deviceRange(0, 0).foldLeft(0.0f)((sum, i) => sum + warpReduceSum(-1, p._1(0)))", "directly initialize")

  test("host calls fail on warp reduction markers"):
    intercept[IllegalStateException](ScalaKernel.warpReduceSum(-1, 1.0f))
    intercept[IllegalStateException](ScalaKernel.warpReduceSum(-1, 1, 16))
