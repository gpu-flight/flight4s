package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class WarpReductionSuite extends FunSuite:
  test("warp tree reductions expose typed sums and pure custom composition"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      kernel("reduce") {
        val mask = UInt.fromBits(-1)
        val i: Expr[Int] = warp.reduceSum("i", mask, threadIdx.x)
        val f: Expr[Float] = warp.reduceSum("f", mask, literal(1.0f), 8)
        val d: Expr[Double] = warp.reduceTree("d", mask, literal(2.0), 16)(_ * _)
        val u: Expr[UInt] = warp.reduceSum("u", mask, literal(UInt.fromBits(1)))
      }
    """), Nil)

  test("width and mask contracts reject incomplete groups before emitting a statement"):
    for width <- Vector(-1, 0, 3, 31, 33, 64) do
      val error = intercept[DslError] {
        kernel("badWidth") { warp.reduceSum("sum", UInt.fromBits(-1), literal(1), width); () }
      }
      assertEquals(error.code, DslErrorCode.InvalidWarpReductionGroup)
      assertNotEquals(error.span, SourceSpan.Unknown)
    for (mask, width) <- Vector((0, 1), (0, 32), (1, 2), (0xffff, 32), (0x55555555, 2), (0xff0, 8)) do
      assertEquals(intercept[DslError] {
        kernel("badMask") { warp.reduceSum("sum", UInt.fromBits(mask), literal(1), width); () }
      }.code, DslErrorCode.InvalidWarpReductionGroup)
    for (mask, width) <- Vector((-1, 32), (0xffff, 16), (0xffff0000, 16), (0xff00ff00, 8), (0x55555555, 1)) do
      val definition = kernel("goodMask") { warp.reduceSum("sum", UInt.fromBits(mask), literal(1), width); () }
      assertEquals(KernelValidator.validate(definition).errors, Vector.empty)

  test("one shuffle and one pure callback per tree level with an identity width-one case"):
    for width <- Vector(1, 2, 4, 8, 16, 32) do
      var calls = 0
      val definition = kernel("tree") {
        warp.reduceTree("sum", UInt.fromBits(-1), threadIdx.x, width) { (left, right) =>
          calls += 1
          left + right
        }
        ()
      }
      val depth = Integer.numberOfTrailingZeros(width)
      assertEquals(calls, depth)
      assertEquals(definition.ir.body.statements.count(_.isInstanceOf[WarpShuffle[?, ?]]), depth)
      val cuda = CudaCodegen.generate(definition).toOption.get.cudaSource
      assertEquals(cuda.sliding("::__shfl_xor_sync".length).count(_ == "::__shfl_xor_sync"), depth)
      assert(!cuda.contains("__syncthreads"))
      assert(!cuda.contains("__syncwarp"))

  test("tree source and IR match an explicit ordered two-lane collective"):
    given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
    val out = output[Int]("out")
    val signature = params(out)
    val reduced = kernel("tree", signature) { _ =>
      out(threadIdx.x) := warp.reduceTree("result", UInt.fromBits(-1), threadIdx.x, 2)(_ - _)
    }
    val explicit = kernel("tree", signature) { _ =>
      val current = let("result_input", threadIdx.x)
      val partner = warp.shuffleXor("result_partner_1", literal(UInt.fromBits(-1)), current, literal(1), 2)
      val low = ((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & literal(1)) === literal(0)
      val result = let("result", choose(low)(current)(partner) - choose(low)(partner)(current))
      out(threadIdx.x) := result
    }
    assertEquals(reduced.ir, explicit.ir)
    assertEquals(CudaCodegen.generate(reduced), CudaCodegen.generate(explicit))

  test("combines and expression-only callbacks cannot hide collective effects"):
    assertEquals(intercept[DslError] {
      kernel("effectfulCombine") {
        warp.reduceTree("sum", UInt.fromBits(-1), literal(1)) { (left, right) =>
          barrier()
          left + right
        }
        ()
      }
    }.code, DslErrorCode.StatementInsideExpression)
    for width <- Vector(1, 32) do
      assertEquals(intercept[DslError] {
        kernel("hiddenCollective") {
          let("bad", choose(literal(true))(
            warp.reduceSum("sum", UInt.fromBits(-1), literal(1), width))(literal(0)))
          ()
        }
      }.code, DslErrorCode.StatementInsideExpression)

  test("unsupported data types dynamic masks and mismatched combine types do not compile"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.reduceTree("x", UInt.fromBits(-1), literal(true))(_ && _); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.reduceSum("x", literal(UInt.fromBits(-1)), literal(1)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.reduceTree("x", UInt.fromBits(-1), literal(1))((a,b) => literal(1.0)); () }
    """).nonEmpty)

  test("generated temporaries remain ordinary scoped bindings and call spans survive lowering"):
    val span = SourceSpan("Reduction.scala", 7, 2, 7, 60)
    given DslSourcePosition = DslSourcePosition(span)
    val definition = kernel("scope") {
      scoped { warp.reduceSum("sum", UInt.fromBits(-1), literal(1)); () }
      scoped { warp.reduceSum("sum", UInt.fromBits(-1), literal(2)); () }
    }
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
    assert(CudaCodegen.generate(definition).toOption.get.sourceMap.entries.exists(_.sourceSpan == span))
    val duplicate = kernel("duplicate") {
      let("sum_input", literal(0))
      warp.reduceSum("sum", UInt.fromBits(-1), literal(1))
      ()
    }
    assert(KernelValidator.validate(duplicate).errors.nonEmpty)

  test("reductions retain the conservative participation warning under a varying guard"):
    val definition = kernel("guarded") {
      when(threadIdx.x < literal(16)) {
        warp.reduceSum("sum", UInt.fromBits(0xffff), threadIdx.x, 16)
        ()
      }
    }
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code),
      Vector.fill(4)(ValidationWarningCode.WarpParticipationMayDiverge))
