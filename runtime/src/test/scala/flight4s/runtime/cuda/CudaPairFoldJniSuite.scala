package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaPairFoldJniSuite extends FunSuite:
  test("pair folds preserve simultaneous state mixed types and conditional updates on CUDA"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val rows = 259
    val columns = 20
    val lengths = Array.tabulate(rows)(row => row % 23 - 2)
    val values = Array.tabulate(rows * columns)(i => (i % 9 - 4).toFloat)
    val definition = kernel("pairFolds", params(
      input[Int]("lengths"), input[Float]("values"), output[Int]("integers"), output[Float]("floats")
    )) { bindings =>
      val (lengths, values, integers, floats) = bindings
      val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
      when(row < literal(rows)) {
        val n = let("n", lengths(row).read)
        val (a, b) = gpuRange("i", literal(0), n)
          .foldLeft("fib", (literal(0), literal(1))) { (s, _) => (s._2, s._1 + s._2) }
        integers(row) := a
        integers(literal(rows) + row) := b
        val elements = gpuRange("j", literal(0), n).map(j => values(row * literal(columns) + j).read)
        val (count, positiveSum) = elements.filter(_ > literal(0.0f))
          .foldLeft("positive", (literal(0), literal(0.0f))) { (s, x) =>
            (s._1 + literal(1), s._2 + x)
          }
        integers(literal(rows * 2) + row) := count
        floats(row) := positiveSum
        val (sum, squares) = elements.foldLeft("moments", (literal(0.0f), literal(0.0f))) {
          (s, x) => (s._1 + x, s._2 + x * x)
        }
        floats(literal(rows) + row) := sum
        floats(literal(rows * 2) + row) := squares
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "pair_folds.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val deviceLengths = context.allocate[Int](rows).toOption.get
      val deviceValues = context.allocate[Float](values.length).toOption.get
      val integers = context.allocate[Int](rows * 3).toOption.get
      val floats = context.allocate[Float](rows * 3).toOption.get
      assertEquals(deviceLengths.copyFrom(lengths), Right(()))
      assertEquals(deviceValues.copyFrom(values), Right(()))
      assertEquals(function.launch(
        definition.bind((deviceLengths, deviceValues, integers, floats)),
        LaunchConfig(Grid.x(3), LaunchBlock.x(128))
      ), Right(()))
      assertEquals(context.synchronize(), Right(()))
      val intResults = integers.copyToArray().toOption.get
      val floatResults = floats.copyToArray().toOption.get
      for row <- 0 until rows do
        val n = lengths(row)
        val (a, b) = (0 until n).foldLeft((0, 1)) { (s, _) => (s._2, s._1 + s._2) }
        val elements = values.slice(row * columns, row * columns + math.max(0, n))
        val positive = elements.filter(_ > 0.0f)
        assertEquals(intResults(row), a, s"first state at row $row")
        assertEquals(intResults(rows + row), b, s"second state at row $row")
        assertEquals(intResults(rows * 2 + row), positive.length, s"count at row $row")
        assertEquals(floatResults(row), positive.foldLeft(0.0f)(_ + _), s"positive sum at row $row")
        assertEquals(floatResults(rows + row), elements.foldLeft(0.0f)(_ + _), s"sum at row $row")
        assertEquals(floatResults(rows * 2 + row), elements.foldLeft(0.0f)((sum, x) => sum + x * x),
          s"squares at row $row")
    finally context.close()
