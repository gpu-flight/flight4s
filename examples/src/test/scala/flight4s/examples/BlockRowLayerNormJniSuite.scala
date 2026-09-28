package flight4s.examples

import munit.FunSuite
import flight4s.core.codegen.GeneratedCudaModule
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, NvrtcCompiler}

class BlockRowLayerNormJniSuite extends FunSuite:
  private def requireCuda(): Unit =
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")

  test("cooperative normalization matches decimal references across uneven and empty lanes"):
    requireCuda()
    val widths = Vector(1, 2, 3, 31, 32, 33, 127, 128, 129, 257, 1025, 4097, 100001)
    val shapes = widths.map(columns => (5, columns, 1e-5)) ++
      Vector((129, 257, 1e-5)) ++
      Vector(java.lang.Double.MIN_VALUE, 1e-90, Double.MaxValue).map(epsilon => (5, 33, epsilon))
    var checked = 0
    shapes.foreach { (rows, columns, epsilon) =>
      val values = Array.tabulate(rows * columns) { i =>
        val column = i % columns
        (i / columns) % 5 match
          case 0 => -7.0f
          case 1 => 1000000.0f + (column % 101 - 50) / 8.0f
          case 2 => if column % 2 == 0 then Float.MaxValue else -Float.MaxValue
          case 3 => (column % 5 - 2) * java.lang.Float.MIN_VALUE
          case _ => (column % 29 - 14) / 8.0f
      }
      val gain = Array.tabulate(columns)(i => (i % 5 - 2) / 2.0f)
      val bias = Array.tabulate(columns)(i => (i % 7 - 3) / 4.0f)
      val originals = Vector(values, gain, bias).map(_.map(java.lang.Float.floatToRawIntBits).toVector)
      val expected = values.grouped(columns).flatMap { row =>
        val (mean, variance) = RowStatisticsReference(row)
        row.indices.map(column => (((row(column).toDouble - mean) / math.sqrt(variance + epsilon)) *
          gain(column).toDouble + bias(column).toDouble).toFloat)
      }.toArray
      val actual = BlockRowLayerNorm.run(values, gain, bias, rows, columns, epsilon)
      assertEquals(actual.length, values.length)
      actual.indices.foreach { index =>
        assert(java.lang.Float.isFinite(actual(index)), s"non-finite $rows/$columns/$index")
        val tolerance = 2e-6 * math.max(1.0, math.abs(expected(index).toDouble))
        assert(math.abs(actual(index).toDouble - expected(index).toDouble) <= tolerance,
          s"$rows/$columns/$index eps=$epsilon actual=${actual(index)} expected=${expected(index)}")
      }
      assertEquals(BlockRowLayerNorm.run(values, gain, bias, rows, columns, epsilon).toVector, actual.toVector)
      assertEquals(Vector(values, gain, bias).map(_.map(java.lang.Float.floatToRawIntBits).toVector), originals)
      checked += actual.length
    }
    assertEquals(shapes.size, 17)
    assertEquals(checked, shapes.map((rows, columns, _) => rows * columns).sum)

  test("constant and extreme affine rows retain the serial contract"):
    requireCuda()
    assertEquals(BlockRowLayerNorm.run(Array.fill(4)(java.lang.Float.MIN_VALUE),
      Array(0.0f, 1.0f, -Float.MaxValue, Float.MaxValue), Array(1.0f, -2.0f, 3.0f, -4.0f),
      1, 4, epsilon = java.lang.Double.MIN_VALUE).toVector, Vector(1.0f, -2.0f, 3.0f, -4.0f))
    val overflow = BlockRowLayerNorm.run(Array(-1.0f, -1.0f, -1.0f, 3.0f),
      Array.fill(4)(Float.MaxValue), Array.fill(4)(0.0f), 1, 4)
    assert(overflow.take(3).forall(java.lang.Float.isFinite))
    assertEquals(overflow.last, Float.PositiveInfinity)

  test("exact launch contracts reject wrong blocks and surplus blocks preserve output tails"):
    requireCuda()
    val generated = BlockRowLayerNorm.generated
    val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated))
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "block_layer_norm_guards.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val function = context.load(artifact).toOption.get.function(generated).toOption.get
      val rows = 129
      val count = rows * 2
      val input = context.allocate[Float](count).toOption.get
      val gain = context.allocate[Float](2).toOption.get
      val bias = context.allocate[Float](2).toOption.get
      val out = context.allocate[Float](count + 32).toOption.get
      val stream = context.createStream().toOption.get
      assertEquals(input.copyFrom(Array.tabulate(count)(i => if i % 2 == 0 then 1.0f else 3.0f)), Right(()))
      assertEquals(gain.copyFrom(Array(2.0f, -4.0f)), Right(()))
      assertEquals(bias.copyFrom(Array(0.25f, 1.0f)), Right(()))
      val arguments = BlockRowLayerNorm.definition.bind((input, gain, bias, out, rows, 2, 3.0))
      assertEquals(out.copyFrom(Array.fill(count + 32)(-123.0f)), Right(()))
      for block <- Vector(LaunchBlock.x(64), LaunchBlock.x(256), LaunchBlock.xy(64, 2)) do
        assert(function.launch(arguments, LaunchConfig(Grid.x(rows), block), stream).isLeft)
      assertEquals(out.copyToArray().toOption.get.toVector, Vector.fill(count + 32)(-123.0f))
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(count + 32)(-123.0f)), Right(()))
        val config = LaunchConfig(Grid.x(rows + 2), LaunchBlock.x(128))
        assertEquals(if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        assertEquals(actual.take(count).toVector, Vector.tabulate(count)(i => if i % 2 == 0 then -0.75f else -1.0f))
        assertEquals(actual.takeRight(32).toVector, Vector.fill(32)(-123.0f))
    finally context.close()
