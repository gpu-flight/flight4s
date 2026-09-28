package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.dsl.ProductFoldState
import flight4s.core.ir.Expr
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

private case class DeviceRotation(a: Expr[Int], b: Expr[Int], c: Expr[Int]) derives ProductFoldState
private case class DeviceSummary(count: Expr[Int], sum: Expr[Double], seen: Expr[Boolean]) derives ProductFoldState

class CudaProductFoldJniSuite extends FunSuite:
  test("derived case-class folds execute ordered simultaneous updates on both streams"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      val definition = kernel("productFolds", params(input[Int]("limits"), output[Int]("out"),
        output[Double]("totals"), value[Int]("rows"))) { p =>
        val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
        when(row < p._4) {
          val limit = p._1(row).read
          val rotation = gpuRange("outer", literal(0), limit).by(3)
            .flatMap(i => gpuRange("inner", literal(0), i % literal(5)).map(j => i + j))
            .filter(_ > literal(0)).map(x => literal(120) / x)
            .foldLeft("rotation", DeviceRotation(literal(1), literal(2), literal(3))) { (s, x) =>
              DeviceRotation(s.b, s.c, s.a + x)
            }
          val summary = gpuRange("i", literal(0), limit).by(2)
            .foldLeft("summary", DeviceSummary(literal(0), literal(1.5), literal(false))) { (s, i) =>
              s.copy(count = s.count + literal(1), sum = s.sum + convert.i32ToF64(s.count + i),
                seen = s.seen || (i === literal(4)))
            }
          val offset = row * literal(5)
          p._2(offset) := rotation.a
          p._2(offset + literal(1)) := rotation.b
          p._2(offset + literal(2)) := rotation.c
          p._2(offset + literal(3)) := summary.count
          p._2(offset + literal(4)) := choose(summary.seen)(literal(1))(literal(0))
          p._3(row) := summary.sum
        }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
        generated.compilerOptions, Vector(generated)), context.computeCapability, "product_folds.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val function = context.load(artifact).toOption.get.function(generated).toOption.get
      val counts = Array.tabulate(129)(i => Vector(-3, 0, 1, 2, 3, 4, 5, 31, 32, 33, 65, 127, 255)(i % 13))
      val limitBuffer = context.allocate[Int](counts.length).toOption.get
      val out = context.allocate[Int](counts.length * 5 + 32).toOption.get
      val totals = context.allocate[Double](counts.length + 32).toOption.get
      val stream = context.createStream().toOption.get
      assertEquals(limitBuffer.copyFrom(counts), Right(()))
      val expected = counts.toVector.flatMap { n =>
        val rotation = (0 until n by 3).flatMap(i => (0 until i % 5).map(j => i + j))
          .filter(_ > 0).map(120 / _).foldLeft((1, 2, 3))((s, x) => (s._2, s._3, s._1 + x))
        Vector(rotation._1, rotation._2, rotation._3, (0 until n by 2).size, if n > 4 then 1 else 0)
      }
      val expectedTotals = counts.toVector.map(n => (0 until n by 2).zipWithIndex
        .foldLeft(1.5)((sum, pair) => sum + pair._1 + pair._2))
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(counts.length * 5 + 32)(Int.MinValue)), Right(()))
        assertEquals(totals.copyFrom(Array.fill(counts.length + 32)(-123.0)), Right(()))
        val arguments = definition.bind((limitBuffer, out, totals, counts.length))
        val config = LaunchConfig(Grid.x(5), LaunchBlock.x(32))
        assertEquals(if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        assertEquals(out.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(Int.MinValue))
        assertEquals(totals.copyToArray().toOption.get.toVector, expectedTotals ++ Vector.fill(32)(-123.0))
      assertEquals(expected.size + expectedTotals.size, 774)
    finally context.close()
