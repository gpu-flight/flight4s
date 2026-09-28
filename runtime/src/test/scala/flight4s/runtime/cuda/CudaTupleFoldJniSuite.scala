package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaTupleFoldJniSuite extends FunSuite:
  test("tuple folds execute simultaneous updates through filtered and nested traversals"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val definition = kernel("tupleFolds", params(input[Int]("limits"), output[Int]("out"),
        output[Float]("totals"), value[Int]("rows"))) { p =>
        val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
        when(row < p._4) {
          val until = p._1(row).read
          val triple = gpuRange("i", literal(0), until).by(2)
            .filter(i => (i % literal(3)) !== literal(0)).map(_ + literal(1))
            .foldLeft("triple", (literal(1), literal(2), literal(3))) { (s, x) =>
              (s._2, s._3, s._1 + x)
            }
          val quad = gpuRange("outer", literal(0), until).by(3)
            .flatMap(i => gpuRange("inner", literal(0), i % literal(4)).map(j => i + j))
            .filter(x => (x % literal(2)) === literal(0))
            .foldLeft("quad", (literal(1), literal(2), literal(3), literal(4))) { (s, x) =>
              (s._2, s._3, s._4, s._1 + x)
            }
          val single = gpuRange("oneIndex", literal(0), until)
            .foldLeft("single", Tuple1(literal(7)))((s, i) => Tuple1(s._1 + i))
          val mixed = gpuRange("mixedIndex", literal(0), until)
            .foldLeft("mixed", (literal(5), literal(1.5f), literal(false))) { (s, i) =>
              (s._1 + literal(1), s._2 + convert.i32ToF32(i), s._3 || (i === literal(3)))
            }
          val offset = row * literal(10)
          p._2(offset) := triple._1
          p._2(offset + literal(1)) := triple._2
          p._2(offset + literal(2)) := triple._3
          p._2(offset + literal(3)) := quad._1
          p._2(offset + literal(4)) := quad._2
          p._2(offset + literal(5)) := quad._3
          p._2(offset + literal(6)) := quad._4
          p._2(offset + literal(7)) := single._1
          p._2(offset + literal(8)) := mixed._1
          p._2(offset + literal(9)) := choose(mixed._3)(literal(1))(literal(0))
          p._3(row) := mixed._2
        }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "tupleFolds.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      try
        val counts = (Vector(-3, -1) ++ (0 to 65) ++ Vector(127, 255)).toArray
        val limits = context.allocate[Int](counts.length).toOption.get
        val out = context.allocate[Int](counts.length * 10).toOption.get
        val totals = context.allocate[Float](counts.length).toOption.get
        try
          assertEquals(limits.copyFrom(counts), Right(()))
          assertEquals(out.copyFrom(Array.fill(counts.length * 10)(Int.MinValue)), Right(()))
          assertEquals(totals.copyFrom(Array.fill(counts.length)(Float.NaN)), Right(()))
          val function = module.function(generated).toOption.get
          assertEquals(function.launch(definition.bind((limits, out, totals, counts.length)),
            LaunchConfig(Grid.x((counts.length + 31) / 32), LaunchBlock.x(32))), Right(()))
          assertEquals(context.synchronize(), Right(()))
          val expected = counts.toVector.flatMap { n =>
            val triple = (0 until n by 2).filter(_ % 3 != 0).map(_ + 1)
              .foldLeft((1, 2, 3))((s, x) => (s._2, s._3, s._1 + x))
            val quad = (0 until n by 3).flatMap(i => (0 until i % 4).map(j => i + j))
              .filter(_ % 2 == 0).foldLeft((1, 2, 3, 4))((s, x) => (s._2, s._3, s._4, s._1 + x))
            Vector(triple._1, triple._2, triple._3, quad._1, quad._2, quad._3, quad._4,
              7 + (0 until n).sum, 5 + math.max(0, n), if n > 3 then 1 else 0)
          }
          val expectedTotals = counts.toVector.map(n => (0 until n).foldLeft(1.5f)((acc, i) => acc + i.toFloat))
          assertEquals(out.copyToArray().toOption.get.toVector, expected)
          assertEquals(totals.copyToArray().toOption.get.toVector, expectedTotals)
          assertEquals(expected.size + expectedTotals.size, 770)
        finally
          totals.close()
          out.close()
          limits.close()
      finally module.close()
    finally context.close()

  test("23-component tuple folds run without a device tuple representation"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val definition = kernel("largeTuple", params(input[Int]("limits"), output[Int]("out"))) { p =>
        val row = threadIdx.x
        when(row < literal(6)) {
          val initial = (
            literal(0), literal(1), literal(2), literal(3), literal(4), literal(5),
            literal(6), literal(7), literal(8), literal(9), literal(10), literal(11),
            literal(12), literal(13), literal(14), literal(15), literal(16), literal(17),
            literal(18), literal(19), literal(20), literal(21), literal(22)
          )
          val result = gpuRange("i", literal(0), p._1(row).read)
            .foldLeft("state", initial)((s, i) => s.tail :* (s.head + i))
          result.toList.zipWithIndex.foreach { (value, column) =>
            p._2(row * literal(23) + literal(column)) := value
          }
        }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "largeTuple.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      try
        val counts = Array(0, 1, 22, 23, 24, 65)
        val limits = context.allocate[Int](counts.length).toOption.get
        val out = context.allocate[Int](counts.length * 23).toOption.get
        try
          assertEquals(limits.copyFrom(counts), Right(()))
          assertEquals(out.copyFrom(Array.fill(138)(Int.MinValue)), Right(()))
          assertEquals(module.function(generated).toOption.get.launch(definition.bind((limits, out)),
            LaunchConfig(Grid.x(1), LaunchBlock.x(32))), Right(()))
          assertEquals(context.synchronize(), Right(()))
          val expected = counts.toVector.flatMap { n =>
            (0 until n).foldLeft((0 until 23).toVector)((s, i) => s.tail :+ (s.head + i))
          }
          assertEquals(expected.size, 138)
          assertEquals(out.copyToArray().toOption.get.toVector, expected)
        finally
          out.close()
          limits.close()
      finally module.close()
    finally context.close()
