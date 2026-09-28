package flight4s.examples

import munit.FunSuite
import flight4s.core.codegen.GeneratedCudaModule
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, NvrtcCompiler}

class RowLayerNormJniSuite extends FunSuite:
  private def requireCuda(): Unit =
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")

  test("layer normalization matches independent population statistics with affine parameters"):
    requireCuda()
    val sizes = Vector(1, 2, 3, 31, 32, 33, 127, 128, 129, 257, 1025, 4097)
    def valuesFor(rows: Int, columns: Int): Array[Float] = Array.tabulate(rows * columns) { i =>
      val column = i % columns
      (i / columns) % 5 match
        case 0 => -7.0f
        case 1 => 1000000.0f + (column % 101 - 50) / 8.0f
        case 2 => if column % 2 == 0 then Float.MaxValue else -Float.MaxValue
        case 3 => (column % 5 - 2) * java.lang.Float.MIN_VALUE
        case _ => (column % 29 - 14) / 8.0f
    }
    val shapes = sizes.map(columns => (5, columns, 1e-5)) ++
      Vector((129, 33, 1e-5), (257, 1, 1e-5)) ++
      Vector(java.lang.Double.MIN_VALUE, 1e-90, Double.MaxValue).map(epsilon => (5, 33, epsilon))
    var checked = 0
    shapes.foreach { (rows, columns, epsilon) =>
      val values = valuesFor(rows, columns)
      val gain = Array.tabulate(columns)(i => (i % 5 - 2) / 2.0f)
      val bias = Array.tabulate(columns)(i => (i % 7 - 3) / 4.0f)
      val originals = Vector(values, gain, bias).map(_.map(java.lang.Float.floatToRawIntBits).toVector)
      val expected = values.grouped(columns).flatMap { row =>
        val (mean, variance) = RowStatisticsReference(row)
        val denominator = math.sqrt(variance + epsilon)
        row.indices.map(column =>
          (((row(column).toDouble - mean) / denominator) * gain(column).toDouble + bias(column).toDouble).toFloat)
      }.toArray
      val actual = RowLayerNorm.run(values, gain, bias, rows, columns, epsilon)
      assertEquals(actual.length, values.length)
      actual.indices.foreach { index =>
        assert(java.lang.Float.isFinite(actual(index)), s"non-finite result at $rows/$columns/$index")
        val tolerance = 2e-6 * math.max(1.0, math.abs(expected(index).toDouble))
        assert(math.abs(actual(index).toDouble - expected(index).toDouble) <= tolerance,
          s"$rows/$columns/$index eps=$epsilon actual=${actual(index)} expected=${expected(index)}")
      }
      assertEquals(RowLayerNorm.run(values, gain, bias, rows, columns, epsilon).toVector, actual.toVector)
      assertEquals(Vector(values, gain, bias).map(_.map(java.lang.Float.floatToRawIntBits).toVector), originals)
      checked += actual.length
    }
    assertEquals(shapes.size, 17)
    assertEquals(checked, 34334)

  test("population variance epsilon placement and extreme affine output have explicit semantics"):
    requireCuda()
    val exact = RowLayerNorm.run(Array(1.0f, 3.0f), Array(2.0f, -4.0f), Array(0.25f, 1.0f),
      1, 2, epsilon = 3.0)
    assertEquals(exact.toVector, Vector(-0.75f, -1.0f))
    val constant = RowLayerNorm.run(Array.fill(4)(java.lang.Float.MIN_VALUE),
      Array(0.0f, 1.0f, -Float.MaxValue, Float.MaxValue), Array(1.0f, -2.0f, 3.0f, -4.0f),
      1, 4, epsilon = java.lang.Double.MIN_VALUE)
    assertEquals(constant.toVector, Vector(1.0f, -2.0f, 3.0f, -4.0f))
    val extreme = RowLayerNorm.run(Array(-1.0f, -1.0f, -1.0f, 3.0f),
      Array.fill(4)(Float.MaxValue), Array.fill(4)(0.0f), 1, 4)
    assert(extreme.take(3).forall(java.lang.Float.isFinite))
    assertEquals(extreme.last, Float.PositiveInfinity)

  test("one compiled layer-normalization kernel handles both stream paths and tail guards"):
    requireCuda()
    val generated = RowLayerNorm.generated
    val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated))
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "layer_norm_guards.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      val function = module.function(generated).toOption.get
      val rows = 129
      val columns = 2
      val count = rows * columns
      val input = context.allocate[Float](count).toOption.get
      val gain = context.allocate[Float](columns).toOption.get
      val bias = context.allocate[Float](columns).toOption.get
      val out = context.allocate[Float](count + 32).toOption.get
      val stream = context.createStream().toOption.get
      assertEquals(input.copyFrom(Array.tabulate(count)(i => if i % 2 == 0 then 1.0f else 3.0f)), Right(()))
      assertEquals(gain.copyFrom(Array(2.0f, -4.0f)), Right(()))
      assertEquals(bias.copyFrom(Array(0.25f, 1.0f)), Right(()))
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(count + 32)(-123.0f)), Right(()))
        val arguments = RowLayerNorm.definition.bind((input, gain, bias, out, rows, columns, 3.0))
        val config = LaunchConfig(Grid.x(2), LaunchBlock.x(128))
        assertEquals(if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        assertEquals(actual.take(count).toVector, Vector.tabulate(count)(i => if i % 2 == 0 then -0.75f else -1.0f))
        assertEquals(actual.takeRight(32).toVector, Vector.fill(32)(-123.0f))
    finally context.close()
