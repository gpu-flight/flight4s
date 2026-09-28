package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaFloatWideningJniSuite extends FunSuite:
  test("Float widening preserves values and enables Double arithmetic on both stream paths"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val boundaries = Vector(0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
      0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x3f800001, 0xbf800001,
      0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
      0x7f800001, 0xff800001)
    val random = new scala.util.Random(42L)
    val values = (boundaries ++ Vector.fill(1023)(random.nextInt())).map(java.lang.Float.intBitsToFloat).toArray
    val count = values.length
    val definition = kernel("widenFloat", params(input[Float]("values"), output[Double]("out"),
      value[Int]("n"), value[Float]("scalar"))) { p =>
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < p._3) {
        val wide = let("wide", convert.f32ToF64(p._1(index).read))
        p._2(index * literal(3)) := wide
        p._2(index * literal(3) + literal(1)) := wide * wide
        p._2(index * literal(3) + literal(2)) := convert.f32ToF64(p._4)
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "float_widening.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      val function = module.function(generated).toOption.get
      val input = context.allocate[Float](count).toOption.get
      val out = context.allocate[Double](count * 3 + 32).toOption.get
      val stream = context.createStream().toOption.get
      assertEquals(input.copyFrom(values), Right(()))
      val config = LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))
      for explicit <- Vector(false, true) do
        val scalar = if explicit then -0.0f else 0.0f
        assertEquals(out.copyFrom(Array.fill(count * 3 + 32)(-123.0)), Right(()))
        val arguments = definition.bind((input, out, count, scalar))
        val launched = if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config)
        assertEquals(launched, Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        values.zipWithIndex.foreach { (value, index) =>
          val wide = value.toDouble
          val expected = Vector(wide, wide * wide, scalar.toDouble)
          expected.zipWithIndex.foreach { (reference, slot) =>
            val result = actual(index * 3 + slot)
            if reference.isNaN then assert(result.isNaN, s"NaN classification at $index/$slot")
            else assertEquals(java.lang.Double.doubleToRawLongBits(result),
              java.lang.Double.doubleToRawLongBits(reference), s"$index/$slot explicit=$explicit")
          }
        }
        assertEquals(actual.takeRight(32).toVector, Vector.fill(32)(-123.0))
      assertEquals(count * 3 * 2, 6258)
    finally context.close()
