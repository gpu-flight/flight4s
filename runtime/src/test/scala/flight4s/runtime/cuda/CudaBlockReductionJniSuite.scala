package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CompilerOptions, CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Expr
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaBlockReductionJniSuite extends FunSuite:
  test("block trees preserve operand order and captured results when scratch is reused in loops"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      val integers = Array.tabulate(3072)(i => i % 17 - 8)
      val unsigned = Array.tabulate(3072)(i => UInt.fromBits(0xfffffff0 + i % 23))
      val floats = Array.tabulate(3072)(i => Vector(1.0e20f, 1.0f, -1.0e20f, 2.0f, -0.0f, 0.0f, 1.25f, -0.5f)(i % 8))
      val doubles = Array.tabulate(3072)(i => Vector(1.0e100, 1.0, -1.0e100, 2.0, -0.0, 0.0, 1.25, -0.5)(i % 8))
      val shapes = Vector(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024).map(LaunchBlock.x) ++
        Vector(LaunchBlock.xy(8, 8), LaunchBlock.xyz(4, 4, 4))
      for shape <- shapes do
        check(context, integers, (a: Int, b: Int) => a + b, (a: Int, b: Int) => a - b,
          (x: Int) => x.toLong, 0, -999, shape)
        check(context, unsigned, (a: UInt, b: UInt) => UInt.fromBits(a.toIntBits + b.toIntBits),
          (a: UInt, b: UInt) => UInt.fromBits(a.toIntBits - b.toIntBits),
          (x: UInt) => x.toIntBits.toLong, UInt.fromBits(0), UInt.fromBits(12345), shape)
        check(context, floats, (a: Float, b: Float) => a + b, (a: Float, b: Float) => a - b,
          (x: Float) => java.lang.Float.floatToRawIntBits(x).toLong, 0.0f, -999.0f, shape)
        check(context, doubles, (a: Double, b: Double) => a + b, (a: Double, b: Double) => a - b,
          java.lang.Double.doubleToRawLongBits, 0.0, -999.0, shape)
    finally context.close()

  test("block trees support Boolean composition and explicit low-precision arithmetic"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      checkScalar(context, rank => rank < literal(127), (a: Expr[Boolean], b: Expr[Boolean]) => a && b, false)
      checkScalar(context, _ => literal(Float16.fromBits(0x3c00.toShort)),
        (a: Expr[Float16], b: Expr[Float16]) => a + b, Float16.fromBits(0x5800.toShort))
      checkScalar(context, _ => literal(BFloat16.fromBits(0x3f80.toShort)),
        (a: Expr[BFloat16], b: Expr[BFloat16]) => a + b, BFloat16.fromBits(0x4300.toShort))
    finally context.close()

  private def checkScalar[T](context: CudaContext, initial: Expr[Int] => Expr[T],
      combine: (Expr[T], Expr[T]) => Expr[T], expected: T)(using
      CudaType[T], CudaHostCodec[T], scala.reflect.ClassTag[T]
  ): Unit =
    val cudaRoot = sys.env.getOrElse("CUDA_PATH", fail("set CUDA_PATH to the toolkit root for low-precision headers"))
    val include = java.nio.file.Path.of(cudaRoot, "include")
    assert(java.nio.file.Files.isRegularFile(include.resolve("cuda_fp16.h")))
    val options = CompilerOptions(additionalNvrtcOptions = Vector(s"--include-path=$include"))
    val definition = kernel("scalarTree", params(output[T]("out"))) { p =>
      val reduction = block.reduction[T]("scratch", LaunchBlock.x(128))
      val result = reduction.reduceTree("result", initial(threadIdx.x))(combine)
      p._1(threadIdx.x) := result
    }
    val generated = CudaCodegen.generate(definition, options).fold(error => fail(error.message), identity)
    val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated)), context.computeCapability, "scalar_tree.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).toOption.get
    val out = context.allocate[T](128).toOption.get
    try
      val function = module.function(generated).toOption.get
      assertEquals(function.launch(definition.bind(Tuple1(out)), LaunchConfig(Grid.x(1), LaunchBlock.x(128))), Right(()))
      assertEquals(context.synchronize(), Right(()))
      assertEquals(out.copyToArray().toOption.get.toVector, Vector.fill(128)(expected))
    finally
      out.close()
      module.close()

  private def tree[T](values: Vector[T], combine: (T, T) => T): T =
    if values.size == 1 then values.head
    else tree(values.grouped(2).map(pair => combine(pair(0), pair(1))).toVector, combine)

  private def check[T](context: CudaContext, data: Array[T], add: (T, T) => T,
      subtract: (T, T) => T, bits: T => Long, zero: T, sentinel: T, shape: LaunchBlock)(using
      AdditiveType[T], CudaHostCodec[T], scala.reflect.ClassTag[T]
  ): Unit =
    val blockSize = shape.x * shape.y * shape.z
    val count = blockSize * 3
    val definition = kernel("blockTree", params(input[T]("source"), output[T]("out"), value[Int]("limit"))) { p =>
      val reduction = block.reduction[T]("scratch", shape)
      val rank = let("rank", threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
      val index = let("index", blockIdx.x * literal(blockSize) + rank)
      gpuFor("iteration", literal(0), literal(3)) { iteration =>
        val item = choose(index < p._3)(p._1(index).read)(literal(zero))
        val sum = reduction.sum("sum", item)
        val difference = reduction.reduceTree("difference", item)(_ - _)
        // Read the first snapshot only after another tree has overwritten the same scratch.
        p._2(iteration * literal(count * 2) + index) := sum
        p._2(iteration * literal(count * 2) + literal(count) + index) := difference
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated)), context.computeCapability, "block_tree.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).toOption.get
    val stream = context.createStream().toOption.get
    val limit = math.max(0, count - 5)
    val source = context.allocate[T](math.max(1, limit)).toOption.get
    val out = context.allocate[T](count * 6 + 16).toOption.get
    try
      val function = module.function(generated).toOption.get
      assertEquals(source.copyFrom(data.take(math.max(1, limit))), Right(()))
      val expected = Vector.tabulate(count * 6 + 16) { index =>
        if index >= count * 6 then sentinel
        else
          val start = (index % count) / blockSize * blockSize
          val inputs = Vector.tabulate(blockSize)(i => if start + i < limit then data(start + i) else zero)
          tree(inputs, if index / count % 2 == 0 then add else subtract)
      }
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(count * 6 + 16)(sentinel)), Right(()))
        val args = definition.bind((source, out, limit))
        val wrongShape = if shape == LaunchBlock.x(1) then LaunchBlock.x(2) else LaunchBlock.x(1)
        val rejected = if explicit then function.launch(args, LaunchConfig(Grid.x(3), wrongShape), stream)
          else function.launch(args, LaunchConfig(Grid.x(3), wrongShape))
        assertEquals(rejected, Left(CudaLaunchFailure.BlockShapeMismatch("blockTree", shape, wrongShape)))
        assert(out.copyToArray().toOption.get.forall(x => bits(x) == bits(sentinel)))
        val config = LaunchConfig(Grid.x(3), shape)
        assertEquals(if explicit then function.launch(args, config, stream) else function.launch(args, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        for i <- actual.indices do
          assertEquals(bits(actual(i)), bits(expected(i)), s"shape=$shape stream=$explicit index=$i")
    finally
      out.close()
      source.close()
      stream.close()
      module.close()
