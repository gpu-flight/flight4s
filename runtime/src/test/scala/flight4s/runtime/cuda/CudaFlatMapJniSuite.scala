package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaFlatMapJniSuite extends FunSuite:
  test("nested Scala traversals preserve guards ordered folds and untouched outputs on CUDA"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val rows = 259
    val width = 5
    val lengths = Array.tabulate(rows)(row => row % 8 - 2)
    val sentinel = -999
    val definition = kernel("nestedTraversals", params(
      input[Int]("lengths"), output[Int]("results"), output[Int]("written")
    )) { bindings =>
      val (lengths, results, written) = bindings
      val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
      when(row < literal(rows)) {
        val n = let("n", lengths(row).read)
        val indices = for
          i <- gpuRange("i", literal(0), n)
          if i > literal(0)
          j <- gpuRange("j", literal(0), i)
          if j !== literal(1)
        yield i * literal(width) + j
        val recurrence = indices.foldLeft("ordered", literal(7))((acc, x) => acc * literal(3) + x)
        results(row) := recurrence
        val (a, b) = indices.foldLeft("pair", (literal(0), literal(1)))((s, x) => (s._2, s._1 + x))
        results(literal(rows) + row) := a
        results(literal(rows * 2) + row) := b
        val expanded = indices.flatMap(x => gpuRange("k", literal(0), literal(2)).map(k => x + k))
          .filter(_ > literal(6)).foldLeft("expanded", literal(0))(_ + _)
        results(literal(rows * 3) + row) := expanded
        indices.foreach(index => written(row * literal(width * width) + index) := index)
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "nested_traversals.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val deviceLengths = context.allocate[Int](rows).toOption.get
      val results = context.allocate[Int](rows * 4).toOption.get
      val written = context.allocate[Int](rows * width * width).toOption.get
      assertEquals(deviceLengths.copyFrom(lengths), Right(()))
      assertEquals(written.copyFrom(Array.fill(rows * width * width)(sentinel)), Right(()))
      assertEquals(function.launch(
        definition.bind((deviceLengths, results, written)), LaunchConfig(Grid.x(3), LaunchBlock.x(128))
      ), Right(()))
      assertEquals(context.synchronize(), Right(()))
      val actual = results.copyToArray().toOption.get
      val actualWritten = written.copyToArray().toOption.get
      for row <- 0 until rows do
        val indices = (for i <- 0 until lengths(row) if i > 0; j <- 0 until i if j != 1
          yield i * width + j).toVector
        val recurrence = indices.foldLeft(7)((acc, x) => acc * 3 + x)
        val (a, b) = indices.foldLeft((0, 1))((s, x) => (s._2, s._1 + x))
        val expanded = indices.flatMap(x => (0 until 2).map(k => x + k)).filter(_ > 6).sum
        assertEquals(actual(row), recurrence, s"ordered fold at row $row")
        assertEquals(actual(rows + row), a, s"pair first at row $row")
        assertEquals(actual(rows * 2 + row), b, s"pair second at row $row")
        assertEquals(actual(rows * 3 + row), expanded, s"third generator at row $row")
        for index <- 0 until width * width do
          assertEquals(actualWritten(row * width * width + index),
            if indices.contains(index) then index else sentinel, s"row $row, index $index")
    finally context.close()
