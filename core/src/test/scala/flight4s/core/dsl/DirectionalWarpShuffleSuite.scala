package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class DirectionalWarpShuffleSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
  private val full = literal(UInt.fromBits(-1))

  test("directional shuffles accept CUDA's unsigned delta and signed lane-mask selectors"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      kernel("directions") {
        val mask = literal(UInt.fromBits(-1))
        val up: Expr[Int] = warp.shuffleUp("up", mask, threadIdx.x, literal(UInt.fromBits(1)))
        val down: Expr[Float] = warp.shuffleDown("down", mask, literal(1.0f), literal(UInt.fromBits(2)), 8)
        val xor: Expr[Double] = warp.shuffleXor("xor", mask, literal(1.0), literal(1), 16)
      }
    """), Nil)

  test("selectors are typed in both the DSL and the operator IR"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.shuffleUp("x", literal(UInt.fromBits(-1)), literal(1), literal(1)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.shuffleDown("x", literal(UInt.fromBits(-1)), literal(1), literal(1)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("bad") { warp.shuffleXor("x", literal(UInt.fromBits(-1)), literal(1), literal(UInt.fromBits(1))); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.*
      import flight4s.core.types.*
      WarpShuffle(LocalVariable("x", I32), literal(UInt.fromBits(-1)), literal(1),
        WarpShuffleOperator.Up, literal(1), 32, I32)
    """).nonEmpty)

  test("each direction emits its qualified native overload once and preserves source locations"):
    val span = SourceSpan("Directions.scala", 10, 1, 10, 50)
    given DslSourcePosition = DslSourcePosition(span)
    val out = output[Float]("out")
    val definition = kernel("directions", params(out)) { _ =>
      val up = warp.shuffleUp("up", full, literal(2.0f), literal(UInt.fromBits(1)), 8)
      val down = warp.shuffleDown("down", full, up, literal(UInt.fromBits(2)), 8)
      val xor = warp.shuffleXor("xorValue", full, down, literal(4), 8)
      out(threadIdx.x) := xor
      out(threadIdx.x + literal(32)) := xor
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    for name <- Vector("__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync") do
      assertEquals(generated.cudaSource.sliding(name.length).count(_ == name), 1)
      assert(generated.cudaSource.contains(s"::$name("))
    assert(generated.cudaSource.contains("::__shfl_up_sync(0xffffffffu, 0x1.0p1f, 0x00000001u, 8)"))
    assert(generated.sourceMap.entries.exists(_.sourceSpan == span))

  test("directional selector literals are restricted to 0 through 31 including unsigned high bits"):
    for selector <- Vector(Int.MinValue, -1, 0, 1, 7, 16, 31, 32, Int.MaxValue) do
      val definitions = Vector(
        kernel("up") { warp.shuffleUp("x", full, literal(1), literal(UInt.fromBits(selector))); () },
        kernel("down") { warp.shuffleDown("x", full, literal(1), literal(UInt.fromBits(selector))); () },
        kernel("xorValue") { warp.shuffleXor("x", full, literal(1), literal(selector)); () }
      )
      definitions.foreach { definition =>
        val errors = KernelValidator.validate(definition).errors
        assertEquals(errors.map(_.code),
          if selector >= 0 && selector <= 31 then Vector.empty else Vector(ValidationCode.InvalidWarpSelector))
        errors.foreach(error => assert(error.location.endsWith(if definition.name == "xorValue" then ".laneMask" else ".delta")))
      }

  test("all directions share the same width and mask validation"):
    for width <- -1 to 33 do
      val definition = kernel("width") {
        warp.shuffleUp("up", full, literal(1), literal(UInt.fromBits(0)), width)
        warp.shuffleDown("down", full, literal(1), literal(UInt.fromBits(0)), width)
        warp.shuffleXor("xorValue", full, literal(1), literal(0), width)
        ()
      }
      val expected = if Set(1, 2, 4, 8, 16, 32).contains(width) then Vector.empty
        else Vector.fill(3)(ValidationCode.InvalidWarpWidth)
      assertEquals(KernelValidator.validate(definition).errors.map(_.code), expected)
    val empty = kernel("empty") {
      warp.shuffleDown("x", literal(UInt.fromBits(0)), literal(1), literal(UInt.fromBits(1)))
      ()
    }
    assertEquals(KernelValidator.validate(empty).errors.map(_.code), Vector(ValidationCode.EmptyWarpMask))

  test("forged selectors report type errors instead of casting boxed values prematurely"):
    val up = WarpShuffle(LocalVariable("up", I32), full, literal(1), WarpShuffleOperator.Up,
      literal(1.0f).asInstanceOf[Expr[UInt]], 32, I32)
    val xor = WarpShuffle(LocalVariable("xorValue", I32), full, literal(1), WarpShuffleOperator.Xor,
      literal(UInt.fromBits(1)).asInstanceOf[Expr[Int]], 32, I32)
    val definition = kernel("forged") {}.ir.copy(body = Block(Vector(up, xor)))
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector.fill(2)(ValidationCode.ExpressionTypeMismatch))
    assert(CudaCodegen.generateModule(CudaModuleIR(Vector.empty, Vector(definition))).isLeft)

  test("all directional operations retain collective effects and their operand reads"):
    val masks = input[UInt]("masks")
    val definition = kernel("effects", params(masks)) { _ =>
      val deltas = sharedArray[UInt]("deltas", 32)
      warp.shuffleDown("down", masks(threadIdx.x).read, threadIdx.x, deltas(threadIdx.x).read)
      warp.shuffleUp("up", masks(threadIdx.x).read, threadIdx.x, deltas(threadIdx.x).read)
      warp.shuffleXor("xorValue", masks(threadIdx.x).read, threadIdx.x, threadIdx.x)
      ()
    }
    val effects = EffectAnalysis.block(definition.ir.body)
    assertEquals(effects.readSpaces, Set(EffectMemorySpace.Global, EffectMemorySpace.Shared))
    assertEquals(effects.writtenSpaces, Set(EffectMemorySpace.Local))
    assert(effects.hasWarpCollective && !effects.hasBarrier && !effects.isPure)

  test("normalization retains identity shuffles and never substitutes the input for the result"):
    val definition = kernel("normalize") {
      val up = warp.shuffleUp("up", full, literal(1) + literal(2), literal(UInt.fromBits(0)), 1)
      val down = warp.shuffleDown("down", full, up, literal(UInt.fromBits(0)), 1)
      val xor = warp.shuffleXor("xorValue", full, down, literal(1) - literal(1), 1)
      local("repeated", xor + xor)
      ()
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.size, 4)
    val operators: Vector[WarpShuffleOperator[?]] = normalized.body.statements.take(3)
      .map(_.asInstanceOf[WarpShuffle[?, ?]].operator)
    assertEquals(operators, Vector[WarpShuffleOperator[?]](
      WarpShuffleOperator.Up, WarpShuffleOperator.Down, WarpShuffleOperator.Xor))
    assertEquals(normalized.body.statements.last, definition.ir.body.statements.last)
    assertEquals(IrNormalizer.kernel(normalized), normalized)
    assert(KernelValidator.validate(normalized).isValid)

  test("directional results retain lexical scope and conservative varying classification"):
    var escaped: Expr[Int] = literal(0)
    val definition = kernel("scope") {
      scoped { escaped = warp.shuffleUp("up", full, literal(1), literal(UInt.fromBits(0))) }
      local("bad", escaped)
      ()
    }
    assertEquals(KernelValidator.validate(definition).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val varying = kernel("varying") {
      val result = warp.shuffleXor("result", full, literal(1), literal(0))
      when(result > literal(0)) { barrier() }
    }
    assertEquals(KernelValidator.validate(varying).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))

  test("none of the directional shuffles can be hidden inside a pure callback"):
    val callbacks = Vector[BlockBuilder ?=> Expr[Int]](
      warp.shuffleUp("up", full, literal(1), literal(UInt.fromBits(1))),
      warp.shuffleDown("down", full, literal(1), literal(UInt.fromBits(1))),
      warp.shuffleXor("xorValue", full, literal(1), literal(1))
    )
    callbacks.foreach { callback =>
      val failure = intercept[DslError] {
        kernel("callback") { choose(literal(true))(callback)(literal(0)); () }
      }
      assertEquals(failure.code, DslErrorCode.StatementInsideExpression)
    }

  test("directional shuffles support dynamic selectors and UInt data"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.UInt
      kernel("dynamic", params(value[UInt]("delta"), value[Int]("laneMask"))) { p =>
        val (delta, laneMask) = p
        val mask = literal(UInt.fromBits(-1))
        val up: Expr[UInt] = warp.shuffleUp("up", mask, mask, delta)
        val down: Expr[UInt] = warp.shuffleDown("down", mask, up, delta)
        val xor: Expr[UInt] = warp.shuffleXor("xor", mask, down, laneMask)
      }
    """), Nil)
