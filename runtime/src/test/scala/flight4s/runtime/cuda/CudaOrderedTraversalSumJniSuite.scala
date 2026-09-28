package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CompilerOptions, CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaOrderedTraversalSumJniSuite extends FunSuite:
  test("ordered sums retain integer wrapping and floating-point order across guarded nested and empty ranges"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      check(context, Array.tabulate(128)(i => i % 17 - 8), (x: Int) => x, 7, -999,
        (a: Int, b: Int) => a + b, (x: Int) => x.toLong)
      check(context, Array.tabulate(128)(i => UInt.fromBits(0xfffffff0 + i % 23)), (x: UInt) => x,
        UInt.fromBits(7), UInt.fromBits(12345),
        (a: UInt, b: UInt) => UInt.fromBits(a.toIntBits + b.toIntBits), (x: UInt) => x.toIntBits.toLong)
      check(context, Array.tabulate(128)(i => Vector(1.0e20f, 1.0f, -1.0e20f, 2.0f, -0.0f, 0.0f,
        Float.PositiveInfinity, Float.NaN)(i % 8)), (x: Float) => x, -0.0f, -999.0f,
        (a: Float, b: Float) => a + b, (x: Float) => java.lang.Float.floatToIntBits(x).toLong)
      check(context, Array.tabulate(128)(i => Vector(1.0e100, 1.0, -1.0e100, 2.0, -0.0, 0.0,
        Double.PositiveInfinity, Double.NaN)(i % 8)), (x: Double) => x, -0.0, -999.0,
        (a: Double, b: Double) => a + b, java.lang.Double.doubleToLongBits)
    finally context.close()

  test("ordered sums promote both half formats and both FP8 formats to float on the GPU"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      def checkLow[T](encodings: Vector[T])(using CudaType[T], AccumulatorType[T, Float], CudaHostCodec[T], scala.reflect.ClassTag[T]): Unit =
        val decoded = Vector(1.0f, -2.0f, 0.5f, 0.0f)
        check(context, Array.tabulate(128)(i => encodings(i % 4)), (x: T) => decoded(encodings.indexOf(x)),
          1.5f, -999.0f, (a: Float, b: Float) => a + b, (x: Float) => java.lang.Float.floatToIntBits(x).toLong)
      checkLow(Vector(0x3c00, 0xc000, 0x3800, 0).map(x => Float16.fromBits(x.toShort)))
      checkLow(Vector(0x3f80, 0xc000, 0x3f00, 0).map(x => BFloat16.fromBits(x.toShort)))
      checkLow(Vector(0x38, 0xc0, 0x30, 0).map(x => Float8E4M3.fromBits(x.toByte)))
      checkLow(Vector(0x3c, 0xc0, 0x38, 0).map(x => Float8E5M2.fromBits(x.toByte)))
    finally context.close()

  private def check[T, A](context: CudaContext, values: Array[T], promote: T => A,
      initial: A, sentinel: A, add: (A, A) => A, bits: A => Long)(using
      CudaType[T], AccumulatorType[T, A], AdditiveType[A],
      CudaHostCodec[T], CudaHostCodec[A], scala.reflect.ClassTag[T], scala.reflect.ClassTag[A]
  ): Unit =
    val include = java.nio.file.Path.of(sys.env.getOrElse("CUDA_PATH", fail("set CUDA_PATH for low-precision headers")), "include")
    val options = CompilerOptions(additionalNvrtcOptions = Vector(s"--include-path=$include"))
    val definition = kernel("orderedSums", params(input[T]("source"), input[Int]("limits"),
      output[A]("out"), value[Int]("rows"))) { p =>
      val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
      when(row < p._4) {
        val limit = p._2(row).read
        val plain = gpuRange("plainIndex", literal(0), limit).map(i => p._1(i).read)
          .sum("plain", literal(initial))
        val filtered = gpuRange("filteredIndex", literal(0), limit).by(2)
          .filter(_ > literal(0)).map(i => p._1(literal(120) / i).read)
          .sum("filtered", literal(initial))
        val nested = gpuRange("outerIndex", literal(0), limit).by(3).filter(_ > literal(0))
          .flatMap(i => gpuRange("innerIndex", literal(0), i % literal(4)).map(_ => p._1(i).read))
          .sum("nested", literal(initial))
        val empty = gpuRange("emptyIndex", literal(0), limit).filter(_ < literal(0))
          .map(i => p._1(i).read).sum("empty", literal(initial))
        val offset = row * literal(5)
        p._3(offset) := plain
        p._3(offset + literal(1)) := filtered
        p._3(offset + literal(2)) := nested
        p._3(offset + literal(3)) := empty
        p._3(offset + literal(4)) := plain
      }
    }
    val generated = CudaCodegen.generate(definition, options).fold(error => fail(error.message), identity)
    val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated)), context.computeCapability, "ordered_sums.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).toOption.get
    val counts = Vector(-3, 0, 1, 2, 3, 4, 5, 31, 32, 33, 65, 127)
    val source = context.allocate[T](values.length).toOption.get
    val limits = context.allocate[Int](counts.size).toOption.get
    val out = context.allocate[A](counts.size * 5 + 16).toOption.get
    val stream = context.createStream().toOption.get
    try
      assertEquals(source.copyFrom(values), Right(()))
      assertEquals(limits.copyFrom(counts.toArray), Right(()))
      val expected = counts.flatMap { n =>
        val plain = (0 until n).foldLeft(initial)((s, i) => add(s, promote(values(i))))
        val filtered = (0 until n by 2).filter(_ > 0).foldLeft(initial)((s, i) => add(s, promote(values(120 / i))))
        val nested = (0 until n by 3).filter(_ > 0).flatMap(i => (0 until i % 4).map(_ => i))
          .foldLeft(initial)((s, i) => add(s, promote(values(i))))
        Vector(plain, filtered, nested, initial, plain)
      } ++ Vector.fill(16)(sentinel)
      val function = module.function(generated).toOption.get
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(counts.size * 5 + 16)(sentinel)), Right(()))
        val args = definition.bind((source, limits, out, counts.size))
        val config = LaunchConfig(Grid.x(1), LaunchBlock.x(32))
        assertEquals(if explicit then function.launch(args, config, stream) else function.launch(args, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        assertEquals(out.copyToArray().toOption.get.toVector.map(bits), expected.map(bits))
    finally
      stream.close()
      out.close()
      limits.close()
      source.close()
      module.close()
