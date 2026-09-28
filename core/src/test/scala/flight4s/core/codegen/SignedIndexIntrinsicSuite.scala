package flight4s.core.codegen

import munit.FunSuite
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Intrinsic, IrNormalizer, KernelValidator, SourceSpan}
import flight4s.core.types.I32

class SignedIndexIntrinsicSuite extends FunSuite:
  test("all nine I32 indexing intrinsics explicitly lower to signed int"):
    for family <- Vector("threadIdx", "blockIdx", "blockDim"); axis <- Vector("x", "y", "z") do
      val span = SourceSpan("Indexes.scala", 8, 3, 8, 16)
      val intrinsic = Intrinsic(s"$family.$axis", I32, span)
      val definition = kernel("indexValue", params(output[Int]("out"))) { p =>
        p._1(literal(0)) := intrinsic
      }
      assertEquals(KernelValidator.validate(definition.ir).errors, Vector.empty)
      assertEquals(IrNormalizer.kernel(definition.ir), definition.ir)
      val generated = CudaCodegen.generate(definition).toOption.get
      assertEquals(generated.cudaSource,
        s"extern \"C\" __global__ void indexValue(int* out) {\n  out[0] = static_cast<int>($family.$axis);\n}\n")

  test("negative operands keep signed comparison division and remainder in CUDA"):
    val definition = kernel("signedIndex", params(output[Int]("out"))) { p =>
      p._1(literal(0)) := choose(literal(-1) < blockDim.x)(literal(1))(literal(0))
      p._1(literal(1)) := literal(-7) / blockDim.x
      p._1(literal(2)) := literal(-7) % blockDim.x
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    assertEquals(generated.cudaSource,
      """extern "C" __global__ void signedIndex(int* out) {
        |  out[0] = ((-1 < static_cast<int>(blockDim.x)) ? 1 : 0);
        |  out[1] = (-7 / static_cast<int>(blockDim.x));
        |  out[2] = (-7 % static_cast<int>(blockDim.x));
        |}
        |""".stripMargin.replace("\r\n", "\n"))
