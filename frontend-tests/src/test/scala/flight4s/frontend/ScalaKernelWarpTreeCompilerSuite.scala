package flight4s.frontend

import munit.FunSuite
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*
import flight4s.core.types.{U32, UInt}

class ScalaKernelWarpTreeCompilerSuite extends FunSuite:
  private def factory(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): String = s"""
    package quotedwarptreefixture
    import scala.annotation.experimental
    import flight4s.frontend.ScalaKernel.*
    $schema
    class Definitions:
      @experimental
      def definition = kernel("tree", params($parameters)) { p =>
        $body
      }
  """

  private def accepted(body: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")")(check: Kernel[Tuple] => Unit): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "WarpTree", factory(body, schema, parameters))
      assertEquals(result.errors, Vector.empty)
      CompilerHarness.withClasses(Seq(result.classes)) { loader =>
        val definition = loader.loadClass("quotedwarptreefixture.Definitions")
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

  private def generated(kernel: Kernel[?]): String =
    CudaCodegen.generate(kernel).toOption.get.cudaSource

  test("full-width Float maxima stage one shuffle per tree level with ordered conditional combines"):
    accepted("""
      val best = warpReduceTree(-1, p._1(threadIdx.x))((a, b) => if a > b then a else b)
      p._2(threadIdx.x) = best
    """) { kernel =>
      assertEquals(shuffles(kernel).size, 5)
      assertEquals(shuffles(kernel).map(maskBits), Vector.fill(5)(-1))
      assertEquals(shuffles(kernel).map(_.width), Vector.fill(5)(32))
      assert(shuffles(kernel).forall(_.span != SourceSpan.Unknown))
      assert(all(kernel.body).forall(_.span != SourceSpan.Unknown))
      val source = generated(kernel)
      assert(!source.contains("__syncthreads"))
      assertEquals(source.sliding("::__shfl_xor_sync".length).count(_ == "::__shfl_xor_sync"), 5)
    }

  test("an addition combine lowers exactly like the dedicated sum marker"):
    var sum = Option.empty[String]
    accepted("val total = warpReduceSum(-1, p._1(threadIdx.x)); p._2(threadIdx.x) = total") { kernel =>
      sum = Some(generated(kernel))
    }
    accepted("val total = warpReduceTree(-1, p._1(threadIdx.x))(_ + _); p._2(threadIdx.x) = total") { kernel =>
      assertEquals(Some(generated(kernel)), sum)
    }

  test("explicit power-of-two widths group masks and the identity width emits no shuffle"):
    accepted("""
      val a = warpReduceTree(0xff00ff00, p._1(threadIdx.x), 8)((a, b) => if a < b then a else b)
      val b = warpReduceTree(-1, p._1(threadIdx.x), 1)(_ + _)
      p._2(threadIdx.x) = a + b
    """) { kernel =>
      assertEquals(shuffles(kernel).map(_.width), Vector.fill(3)(8))
      assertEquals(shuffles(kernel).map(maskBits), Vector.fill(3)(0xff00ff00))
    }

  test("Int and Double combines reduce with vars loops and the library fullMask constant"):
    accepted("""
      var total = 0.0
      for round <- deviceRange(0, p._3) do
        val grouped = warpReduceTree(sync.fullMask, p._1(threadIdx.x), 8)((a, b) => a - b)
        total += grouped
      p._2(threadIdx.x) = total
    """, parameters = "input[Double](\"data\"), output[Double](\"target\"), value[Int](\"count\")") { kernel =>
      assertEquals(shuffles(kernel).map(_.width), Vector.fill(3)(8))
    }
    accepted("val total = warpReduceTree(-1, p._1(threadIdx.x))((a, b) => a | b); p._2(threadIdx.x) = total",
      parameters = "input[Int](\"data\"), output[Int](\"target\"), value[Int](\"count\")") { kernel =>
      assertEquals(shuffles(kernel).size, 5)
    }

  test("qualified and renamed imported tree symbols share the same lowering"):
    accepted("""
      val a = flight4s.frontend.ScalaKernel.warpReduceTree(-1, p._1(0))(_ + _)
      val b = combine(-1, p._1(1))((x, y) => if x > y then x else y)
      p._2(0) = a + b
    """, schema = "import flight4s.frontend.ScalaKernel.{warpReduceTree as combine}") { kernel =>
      assertEquals(shuffles(kernel).size, 10)
    }

  test("divergent enclosing control warns per shuffle while uniform control stays quiet"):
    accepted("if threadIdx.x < 16 then { val total = warpReduceTree(0x0000ffff, p._1(threadIdx.x), 2)(_ + _); p._2(threadIdx.x) = total }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings.map(_.code), Vector(ValidationWarningCode.WarpParticipationMayDiverge))
    }
    accepted("if blockIdx.x == 0 then { val total = warpReduceTree(-1, p._1(threadIdx.x), 2)(_ + _); p._2(threadIdx.x) = total }") { kernel =>
      assertEquals(KernelValidator.validate(kernel).warnings, Vector.empty)
    }

  private def rejected(body: String, message: String, schema: String = "", parameters: String =
      "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\")"): Unit =
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "RejectedWarpTree", factory(body, schema, parameters))
      assert(result.errors.exists(_.contains(message)), result.errors.mkString("\n"))
      assert(!result.errors.exists(_.contains("Exception occurred while executing macro expansion")), result.errors.mkString("\n"))
    }

  test("masks and widths keep the complete aligned compile-time group admission"):
    rejected("val total = warpReduceTree(0, p._1(0))(_ + _); p._2(0) = total", "at least one participating lane")
    rejected("val total = warpReduceTree(p._3, p._1(0))(_ + _); p._2(0) = total", "compile-time constant Int")
    rejected("val total = warpReduceTree(-1, p._1(0), 3)(_ + _); p._2(0) = total", "width")
    rejected("val total = warpReduceTree(-1, p._1(0), p._3)(_ + _); p._2(0) = total", "width")
    rejected("val total = warpReduceTree(0x0000ffff, p._1(0))(_ + _); p._2(0) = total", "complete aligned groups")
    rejected("val total = warpReduceTree(0x55555555, p._1(0), 2)(_ + _); p._2(0) = total", "complete aligned groups")

  test("combines must be literal pure matching lambdas without captures or effects"):
    rejected("val f = (a: Float, b: Float) => a + b; val total = warpReduceTree(-1, p._1(0))(f); p._2(0) = total", "primitive locals")
    rejected("val total = warpReduceTree(-1, p._1(0))({ println(1); (a: Float, b: Float) => a + b }); p._2(0) = total", "combine lambda")
    rejected("val total = warpReduceTree(-1, p._1(0))((a, b) => { p._2(0) = a; a + b }); p._2(0) = total", "expression blocks")
    rejected("val total = warpReduceTree(-1, p._1(0))((a, b) => a + Host.bias); p._2(0) = total", "captures", "object Host { val bias = 1.0f }")
    rejected("val total = warpReduceTree(-1, p._4)((a, b) => a && b); p._2(0) = if total then 1.0f else 0.0f", "Int, Float and Double",
      parameters = "input[Float](\"data\"), output[Float](\"target\"), value[Int](\"count\"), value[Boolean](\"flag\")")

  test("tree reductions are initializer-only and cannot hide hosts or lookalikes"):
    rejected("p._2(0) = warpReduceTree(-1, p._1(0))(_ + _)", "directly initialize")
    rejected("val total = warpReduceTree(-1, p._1(0))(_ + _) + 1.0f; p._2(0) = total", "directly initialize")
    rejected("val unused = deviceRange(0, 0).map(i => { val total = warpReduceTree(-1, p._1(0))(_ + _); i })", "expression blocks")
    rejected("val total = Host.warpReduceTree(-1, p._1(0))((a, b) => a + b); p._2(0) = total", "host/helper calls",
      "object Host { def warpReduceTree(mask: Int, value: Float)(combine: (Float, Float) => Float): Float = { println(1); value } }")

  test("host calls fail on warp tree reduction markers"):
    intercept[IllegalStateException](ScalaKernel.warpReduceTree(-1, 1.0f)(_ + _))
    intercept[IllegalStateException](ScalaKernel.warpReduceTree(-1, 1, 16)((a, b) => if a > b then a else b))
