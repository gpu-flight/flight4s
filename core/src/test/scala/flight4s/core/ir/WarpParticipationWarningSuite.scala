package flight4s.core.ir

import munit.FunSuite
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.UInt
import flight4s.core.codegen.CudaCodegen

class WarpParticipationWarningSuite extends FunSuite:
  private val full = literal(UInt.fromBits(-1))

  test("warp calls under varying control report a participation warning"):
    val definition = kernel("warpGuard") {
      val lane = let("lane", threadIdx.x)
      when(lane < literal(16)) { warp.sync(full) }
    }
    val validation = KernelValidator.validate(definition)
    assertEquals(validation.errors, Vector.empty)
    assertEquals(validation.warnings.map(_.code), Vector(ValidationWarningCode.WarpParticipationMayDiverge))
    assertEquals(validation.warnings.head.location, "body.statements[1].then.statements[0]")
    assertNotEquals(validation.warnings.head.span, SourceSpan.Unknown)

  test("all shuffle vote and sync variants retain participation diagnostics"):
    val definition = kernel("allWarpCalls") {
      when(threadIdx.x < literal(16)) {
        warp.shuffle("direct", full, threadIdx.x, literal(0))
        warp.shuffleUp("up", full, threadIdx.x, literal(UInt.fromBits(1)))
        warp.shuffleDown("down", full, threadIdx.x, literal(UInt.fromBits(1)))
        warp.shuffleXor("xorResult", full, threadIdx.x, literal(1))
        warp.ballot("ballot", full, literal(true))
        warp.all("all", full, literal(true))
        warp.any("any", full, literal(true))
        warp.sync(full)
      }
    }
    val validation = KernelValidator.validate(definition)
    assertEquals(validation.errors, Vector.empty)
    assertEquals(validation.warnings.map(_.code), Vector.fill(8)(ValidationWarningCode.WarpParticipationMayDiverge))
    assertEquals(validation.warnings.map(_.location), Vector.tabulate(8)(i => s"body.statements[0].then.statements[$i]"))
    assert(CudaCodegen.generate(definition).isRight)

  test("mask uniformity follows local loads stores and branch assignments"):
    val masks = input[UInt]("masks")
    val definition = kernel("varyingMasks", params(masks)) { _ =>
      val mask = local("mask", masks(threadIdx.x).read)
      warp.sync(mask.read)
      mask := full
      warp.sync(mask.read)
      when(threadIdx.x < literal(16)) { mask := literal(UInt.fromBits(0xffff)) }
      warp.sync(mask.read)
    }
    val validation = KernelValidator.validate(definition)
    assertEquals(validation.errors, Vector.empty)
    assertEquals(validation.warnings.map(_.location), Vector("body.statements[1]", "body.statements[5]"))
    assert(validation.warnings.forall(_.message.contains("lane-varying mask")))

  test("varying loops branches and lexical scopes propagate control without leaking it"):
    val definition = kernel("nestedWarpControl") {
      gpuIf(threadIdx.x < literal(16)) {
        scoped { warp.sync(full) }
      } { warp.sync(full) }
      gpuFor("i", literal(0), threadIdx.x) { _ => warp.sync(full) }
      warp.sync(full)
    }
    val validation = KernelValidator.validate(definition)
    assertEquals(validation.warnings.map(_.location), Vector(
      "body.statements[0].then.statements[0].body.statements[0]",
      "body.statements[0].else.statements[0]", "body.statements[1].body.statements[0]"))
    assert(validation.warnings.forall(_.message.contains("lane-varying control flow")))

  test("combined varying mask and control produce one actionable warning per call"):
    val masks = input[UInt]("masks")
    val definition = kernel("bothCauses", params(masks)) { _ =>
      when(threadIdx.x < literal(16)) { warp.sync(masks(threadIdx.x).read) }
    }
    val warning = KernelValidator.validate(definition).warnings
    assertEquals(warning.size, 1)
    assert(warning.head.message.contains("lane-varying control flow and a lane-varying mask"))
    assert(warning.head.message.contains("same call with the same mask"))

  test("uniform controls and masks stay quiet without claiming valid lane membership"):
    val mask = value[UInt]("mask")
    val count = value[Int]("count")
    val definition = kernel("uniformWarpControl", params(mask, count)) { _ =>
      val captured = let("captured", mask)
      when(blockIdx.x < count) {
        gpuFor("i", literal(0), count) { _ =>
          warp.sync(captured)
          warp.ballot("selected", captured, threadIdx.x < literal(16))
        }
      }
    }
    val validation = KernelValidator.validate(definition)
    assertEquals(validation.errors, Vector.empty)
    assertEquals(validation.warnings, Vector.empty)

  test("partial masks may warn conservatively and module paths preserve the call location"):
    val definition = kernel("partialMask") {
      when(threadIdx.x < literal(16)) { warp.sync(literal(UInt.fromBits(0xffff))) }
    }
    val validation = ModuleValidator.validate(module(kernels = Vector(definition)))
    assertEquals(validation.errors, Vector.empty)
    assertEquals(validation.warnings.size, 1)
    assert(validation.warnings.head.location.endsWith("body.statements[0].then.statements[0]"))
    val original = definition.ir
    val generated = CudaCodegen.generate(definition).toOption.get
    assertEquals(definition.ir, original)
    assert(generated.cudaSource.contains("::__syncwarp(0x0000ffffu);"))
    assert(!generated.cudaSource.contains("__syncthreads"))

  test("unknown uniformity remains unproven and empty masks still produce hard errors"):
    val unknown = KernelIR("unknown", params(), Block(Vector(
      IfThen(Intrinsic("unknownPredicate", flight4s.core.types.Bool), Block(Vector(WarpBarrier(full)))))))
    assertEquals(KernelValidator.validate(unknown).warnings, Vector.empty)
    assert(KernelValidator.validate(unknown).errors.exists(_.code == ValidationCode.UnknownIntrinsic))
    val empty = kernel("emptyMask") { warp.sync(literal(UInt.fromBits(0))) }
    assertEquals(KernelValidator.validate(empty).errors.map(_.code), Vector(ValidationCode.EmptyWarpMask))
