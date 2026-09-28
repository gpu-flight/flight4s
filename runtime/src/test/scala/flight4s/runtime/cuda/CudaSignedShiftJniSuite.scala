package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaSignedShiftJniSuite extends FunSuite:
  test("signed GPU shifts rotation and functional bit extraction match Scala edge and random values"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try check(context)
    finally context.close()

  private def check(context: CudaContext): Unit =
    val count = 2053
    val random = new scala.util.Random(174)
    val words = Array(0, 1, -1, -2, Int.MinValue, Int.MaxValue, 0x55555555, 0xaaaaaaaa)
    val distances = Array(Int.MinValue, -65, -33, -32, -31, -1, 0, 1, 15, 31, 32, 33, 63, 64, Int.MaxValue)
    val a = Array.tabulate(count)(i => if i < words.length * distances.length then words(i / distances.length) else random.nextInt())
    val b = Array.tabulate(count)(i => if i < words.length * distances.length then distances(i % distances.length) else random.nextInt())
    val definition = kernel("signed_shifts", params(input[Int]("flight4s_value_0"), input[Int]("flight4s_value_1"),
      output[Int]("flight4s_value_2"), value[Int]("count"))) { p =>
      val (words, distances, out, count) = p
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < count) {
        val word = words(index).read
        val distance = distances(index).read
        out(index) := word << distance
        out(count + index) := word >> distance
        out(count * literal(2) + index) := word >>> distance
        val amount = let("amount", distance & literal(31))
        out(count * literal(3) + index) := (word << amount) | (word >>> (literal(32) - amount))
        val ones = gpuRange("bit", literal(0), literal(32))
          .map(bit => (word >>> bit) & literal(1))
          .filter(bit => bit === literal(1)).foldLeft("ones", literal(0))(_ + _)
        out(count * literal(4) + index) := ones
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
    val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "signed_shifts.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).fold(failure => fail(failure.message), identity)
    try
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val wordsBuffer = context.allocate[Int](count).toOption.get
      val countsBuffer = context.allocate[Int](count).toOption.get
      val out = context.allocate[Int](count * 5).toOption.get
      try
        assertEquals(wordsBuffer.copyFrom(a), Right(()))
        assertEquals(countsBuffer.copyFrom(b), Right(()))
        assertEquals(function.launch(definition.bind((wordsBuffer, countsBuffer, out, count)),
          LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))), Right(()))
        assertEquals(context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        for i <- 0 until count do
          val expected = Vector(a(i) << b(i), a(i) >> b(i), a(i) >>> b(i), Integer.rotateLeft(a(i), b(i)), Integer.bitCount(a(i)))
          expected.zipWithIndex.foreach { (value, operation) =>
            assertEquals(actual(operation * count + i), value, s"i=$i operation=$operation value=${a(i)} count=${b(i)}")
          }
      finally
        wordsBuffer.close()
        countsBuffer.close()
        out.close()
    finally module.close()
