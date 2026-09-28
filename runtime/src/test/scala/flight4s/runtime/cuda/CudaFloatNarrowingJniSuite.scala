package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.RoundingMode
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaFloatNarrowingJniSuite extends FunSuite:
  private val modes = Vector(RoundingMode.NearestEven, RoundingMode.TowardZero,
    RoundingMode.TowardNegative, RoundingMode.TowardPositive)

  test("host narrowing reference handles ties, underflow, overflow, and signed special values"):
    val tie = 1.0 + java.lang.Math.scalb(1.0, -24)
    val halfSubnormal = java.lang.Float.MIN_VALUE.toDouble / 2.0
    val cases = Vector(
      tie -> Vector(0x3f800000, 0x3f800000, 0x3f800000, 0x3f800001),
      (1.0 + java.lang.Math.scalb(3.0, -24)) -> Vector(0x3f800002, 0x3f800001, 0x3f800001, 0x3f800002),
      -tie -> Vector(0xbf800000, 0xbf800000, 0xbf800001, 0xbf800000),
      halfSubnormal -> Vector(0, 0, 0, 1),
      -halfSubnormal -> Vector(0x80000000, 0x80000000, 0x80000001, 0x80000000),
      Double.MaxValue -> Vector(0x7f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000),
      -Double.MaxValue -> Vector(0xff800000, 0xff7fffff, 0xff800000, 0xff7fffff),
      0.0 -> Vector.fill(4)(0), -0.0 -> Vector.fill(4)(0x80000000),
      Double.PositiveInfinity -> Vector.fill(4)(0x7f800000),
      Double.NegativeInfinity -> Vector.fill(4)(0xff800000))
    for (value, expected) <- cases do
      assertEquals(modes.map(mode => java.lang.Float.floatToRawIntBits(reference(value, mode))), expected)
    assert(modes.forall(mode => reference(Double.NaN, mode).isNaN))

  test("Double narrowing matches all rounding modes on both stream paths"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val halfSubnormal = java.lang.Float.MIN_VALUE.toDouble / 2.0
    val overflowTie = java.lang.Float.MAX_VALUE.toDouble + java.lang.Math.scalb(1.0, 103)
    val boundaries = Vector(0.0, -0.0, java.lang.Double.MIN_VALUE, -java.lang.Double.MIN_VALUE,
      java.lang.Double.MIN_NORMAL, -java.lang.Double.MIN_NORMAL,
      java.lang.Float.MAX_VALUE.toDouble, -java.lang.Float.MAX_VALUE.toDouble,
      Double.MaxValue, -Double.MaxValue, Double.PositiveInfinity, Double.NegativeInfinity,
      Double.NaN, java.lang.Double.longBitsToDouble(0xfff8000000000000L),
      java.lang.Double.longBitsToDouble(0x7ff0000000000001L),
      java.lang.Double.longBitsToDouble(0xfff0000000000001L))
    val boundaryNeighbors = Vector(halfSubnormal, overflowTie).flatMap { x =>
      Vector(java.lang.Math.nextDown(x), x, java.lang.Math.nextUp(x)).flatMap(v => Vector(v, -v))
    }
    val midpointNeighbors = Vector(0, 1, 0x007fffff, 0x00800000, 0x3f7fffff,
      0x3f800000, 0x3f800001, 0x7f7ffffe).flatMap { bits =>
      val lower = java.lang.Float.intBitsToFloat(bits).toDouble
      val upper = java.lang.Float.intBitsToFloat(bits + 1).toDouble
      val midpoint = (lower + upper) / 2.0
      Vector(lower, java.lang.Math.nextDown(midpoint), midpoint, java.lang.Math.nextUp(midpoint), upper)
        .flatMap(v => Vector(v, -v))
    }
    val random = new scala.util.Random(84L)
    val values = (boundaries ++ boundaryNeighbors ++ midpointNeighbors ++
      Vector.fill(1023)(java.lang.Double.longBitsToDouble(random.nextLong()))).toArray
    val count = values.length
    assertEquals(count, 1131)
    val definition = kernel("narrowDouble", params(input[Double]("values"), output[Float]("out"),
      value[Int]("n"), value[Double]("scalar"))) { p =>
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < p._3) {
        val source = let("source", p._1(index).read)
        modes.zipWithIndex.foreach { (mode, slot) =>
          p._2(index * literal(9) + literal(slot)) := convert.f64ToF32(source, mode)
          p._2(index * literal(9) + literal(slot + 5)) := convert.f64ToF32(p._4, mode)
        }
        p._2(index * literal(9) + literal(4)) := convert.f64ToF32(source)
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated))
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "float_narrowing.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      val function = module.function(generated).toOption.get
      val input = context.allocate[Double](count).toOption.get
      val out = context.allocate[Float](count * 9 + 32).toOption.get
      val stream = context.createStream().toOption.get
      assertEquals(input.copyFrom(values), Right(()))
      val config = LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))
      for explicit <- Vector(false, true) do
        val tie = 1.0 + java.lang.Math.scalb(1.0, -24)
        val scalar = if explicit then -tie else tie
        assertEquals(out.copyFrom(Array.fill(count * 9 + 32)(-123.0f)), Right(()))
        val arguments = definition.bind((input, out, count, scalar))
        val launched = if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config)
        assertEquals(launched, Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        values.zipWithIndex.foreach { (value, index) =>
          val expected = modes.map(reference(value, _)) ++ Vector(value.toFloat) ++ modes.map(reference(scalar, _))
          expected.zipWithIndex.foreach { (expectedValue, slot) =>
            val result = actual(index * 9 + slot)
            if expectedValue.isNaN then assert(result.isNaN, s"NaN classification at $index/$slot")
            else assertEquals(java.lang.Float.floatToRawIntBits(result),
              java.lang.Float.floatToRawIntBits(expectedValue), s"$index/$slot value=$value explicit=$explicit")
          }
        }
        assertEquals(actual.takeRight(32).toVector, Vector.fill(32)(-123.0f))
      assertEquals(count * 9 * 2, 20358)
    finally context.close()

  private def reference(exact: Double, rounding: RoundingMode): Float =
    val nearest = exact.toFloat
    val down = if nearest.toDouble > exact then java.lang.Math.nextDown(nearest) else nearest
    val up = if nearest.toDouble < exact then java.lang.Math.nextUp(nearest) else nearest
    rounding match
      case RoundingMode.NearestEven => nearest
      case RoundingMode.TowardNegative => down
      case RoundingMode.TowardPositive => up
      case RoundingMode.TowardZero => if exact < 0.0 then up else down
