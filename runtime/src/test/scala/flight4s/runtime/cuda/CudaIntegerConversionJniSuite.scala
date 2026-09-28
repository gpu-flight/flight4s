package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.RoundingMode
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.UInt

class CudaIntegerConversionJniSuite extends FunSuite:
  test("integer conversion rounding matches exact references at precision boundaries on CUDA"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val boundaries = Vector(Int.MinValue, Int.MinValue + 1, -16777219, -16777217, -16777216,
      -1, 0, 1, 16777215, 16777216, 16777217, 16777218, 16777219, Int.MaxValue - 1, Int.MaxValue)
    val values = (boundaries ++ Vector.tabulate(244)(i => (i.toLong * 15485863 - 1900000000L).toInt)).toArray
    val count = values.length
    val definition = kernel("integerFloating", params(
      input[Int]("signedInput"), input[UInt]("unsignedInput"), output[Float]("floats"), output[Double]("doubles")
    )) { bindings =>
      val (signedInput, unsignedInput, floats, doubles) = bindings
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < literal(count)) {
        val i = signedInput(index).read
        val u = unsignedInput(index).read
        RoundingMode.values.zipWithIndex.foreach { (mode, slot) =>
          floats(literal(slot * count) + index) := convert.i32ToF32(i, mode)
          floats(literal((slot + 4) * count) + index) := convert.u32ToF32(u, mode)
        }
        doubles(index) := convert.i32ToF64(i)
        doubles(literal(count) + index) := convert.u32ToF64(u)
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(
      generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
    )
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "integer_floating.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val signedInput = context.allocate[Int](count).toOption.get
      val unsignedInput = context.allocate[UInt](count).toOption.get
      val floats = context.allocate[Float](count * 8).toOption.get
      val doubles = context.allocate[Double](count * 2).toOption.get
      assertEquals(signedInput.copyFrom(values), Right(()))
      assertEquals(unsignedInput.copyFrom(values.map(UInt.fromBits)), Right(()))
      assertEquals(function.launch(
        definition.bind((signedInput, unsignedInput, floats, doubles)), LaunchConfig(Grid.x(3), LaunchBlock.x(128))
      ), Right(()))
      assertEquals(context.synchronize(), Right(()))
      val actualFloats = floats.copyToArray().toOption.get
      val actualDoubles = doubles.copyToArray().toOption.get
      for index <- values.indices do
        val signed = values(index).toDouble
        val unsigned = java.lang.Integer.toUnsignedLong(values(index)).toDouble
        RoundingMode.values.zipWithIndex.foreach { (mode, slot) =>
          assertEquals(java.lang.Float.floatToRawIntBits(actualFloats(slot * count + index)),
            java.lang.Float.floatToRawIntBits(reference(signed, mode)), s"signed $signed, $mode")
          assertEquals(java.lang.Float.floatToRawIntBits(actualFloats((slot + 4) * count + index)),
            java.lang.Float.floatToRawIntBits(reference(unsigned, mode)), s"unsigned $unsigned, $mode")
        }
        assertEquals(actualDoubles(index), signed)
        assertEquals(actualDoubles(count + index), unsigned)
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
