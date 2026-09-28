package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaWarpReductionJniSuite extends FunSuite:
  test("ordered warp trees agree in every selected lane across widths masks and multidimensional blocks"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      val integers = Array.tabulate(384)(i => i % 17 - 8)
      val unsigned = Array.tabulate(384)(i => UInt.fromBits(0xfffffff0 + i % 23))
      val floats = Array.tabulate(384)(i => Vector(1.0e20f, 1.0f, -1.0e20f, 2.0f, -0.0f, 0.0f, 1.25f, -0.5f)(i % 8))
      val doubles = Array.tabulate(384)(i => Vector(1.0e100, 1.0, -1.0e100, 2.0, -0.0, 0.0, 1.25, -0.5)(i % 8))
      for width <- Vector(1, 2, 4, 8, 16, 32) do
        val all = UInt.fromBits(-1)
        val alternate = UInt.fromBits((0 until 32 by width).filter(_ / width % 2 == 0)
          .foldLeft(0L)((bits, start) => bits | (((1L << width) - 1L) << start)).toInt)
        val firstGroup = UInt.fromBits(((1L << width) - 1L).toInt)
        for mask <- Vector(all, alternate, firstGroup).distinct do
          check(context, integers, (a: Int, b: Int) => a + b, (a: Int, b: Int) => a - b,
            (x: Int) => x.toLong, -999, width, mask)
          check(context, unsigned, (a: UInt, b: UInt) => UInt.fromBits(a.toIntBits + b.toIntBits),
            (a: UInt, b: UInt) => UInt.fromBits(a.toIntBits - b.toIntBits),
            (x: UInt) => x.toIntBits.toLong, UInt.fromBits(12345), width, mask)
          check(context, floats, (a: Float, b: Float) => a + b, (a: Float, b: Float) => a - b,
            (x: Float) => java.lang.Float.floatToRawIntBits(x).toLong, -999.0f, width, mask)
          check(context, doubles, (a: Double, b: Double) => a + b, (a: Double, b: Double) => a - b,
            java.lang.Double.doubleToRawLongBits, -999.0, width, mask)
    finally context.close()

  private def tree[T](values: Vector[T], combine: (T, T) => T): T =
    if values.size == 1 then values.head
    else tree(values.grouped(2).map(pair => combine(pair(0), pair(1))).toVector, combine)

  private def check[T](context: CudaContext, data: Array[T], add: (T, T) => T,
      subtract: (T, T) => T, bits: T => Long, sentinel: T, width: Int, mask: UInt)(using
      shuffleType: WarpShuffleType[T], additiveType: AdditiveType[T], codec: CudaHostCodec[T],
      classTag: scala.reflect.ClassTag[T]
  ): Unit =
    val definition = kernel("warpTree", params(input[T]("source")(using shuffleType),
      output[T]("out")(using shuffleType), value[Int]("count"))) { p =>
      val linear = let("linear", threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
      val index = let("index", blockIdx.x * blockDim.x * blockDim.y * blockDim.z + linear)
      val lane = linear & literal(31)
      val selected = ((literal(mask) >> lane) & literal(UInt.fromBits(1))) !== literal(UInt.fromBits(0))
      when(selected) {
        val sum = warp.reduceSum("sum", mask, p._1(index).read, width)
        val difference = warp.reduceTree("difference", mask, p._1(index).read, width)(_ - _)
        p._2(index) := sum
        p._2(p._3 + index) := difference
        p._2(p._3 * literal(2) + index) := sum
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated)), context.computeCapability, "warp_tree.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).toOption.get
    val stream = context.createStream().toOption.get
    try
      val function = module.function(generated).toOption.get
      val shapes = Vector(LaunchBlock.x(128), LaunchBlock.xy(8, 8), LaunchBlock.xyz(4, 4, 4)) ++
        Option.when(mask.toIntBits == ((1L << width) - 1L).toInt)(LaunchBlock.x(width)).toVector
      for shape <- shapes do
        val blockSize = shape.x * shape.y * shape.z
        val count = blockSize * 3
        val source = context.allocate[T](count)(using shuffleType).toOption.get
        val out = context.allocate[T](count * 3 + 16)(using shuffleType).toOption.get
        try
          assertEquals(source.copyFrom(data.take(count)), Right(()))
          val expected = Vector.tabulate(count * 3 + 16) { index =>
            val i = index % count
            val lane = (i % blockSize) % 32
            if index >= count * 3 || (mask.toIntBits & (1 << lane)) == 0 then sentinel
            else
              val start = i - (i % blockSize) % width
              val combine = if index / count == 1 then subtract else add
              tree(data.slice(start, start + width).toVector, combine)
          }
          for explicit <- Vector(false, true) do
            assertEquals(out.copyFrom(Array.fill(count * 3 + 16)(sentinel)), Right(()))
            val arguments = definition.bind((source, out, count))
            val config = LaunchConfig(Grid.x(3), shape)
            assertEquals(if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config), Right(()))
            assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
            val actual = out.copyToArray().toOption.get
            for i <- actual.indices do
              assertEquals(bits(actual(i)), bits(expected(i)), s"width=$width mask=$mask shape=$shape stream=$explicit index=$i")
        finally
          source.close()
          out.close()
    finally
      stream.close()
      module.close()
