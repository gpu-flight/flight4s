package flight4s.runtime.cuda

import munit.FunSuite

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Expr
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaMathJniSuite extends FunSuite:
  test("standard Float and Double device math match references including special values"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val values = Vector(Double.NegativeInfinity, -4.0, -1.0, -0.0, 0.0, 0.25, 1.0, 2.0, 4.0,
      Double.PositiveInfinity, Double.NaN) ++ Vector.tabulate(64)(i => (i - 32) / 8.0)
    val count = values.size
    val definition = kernel("standardDeviceMath", params(
      input[Float]("floatInput"), input[Double]("doubleInput"),
      output[Float]("floatOutput"), output[Double]("doubleOutput")
    )) { bindings =>
      val (floatInput, doubleInput, floatOutput, doubleOutput) = bindings
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < literal(count)) {
        // These names must not shadow the globally-qualified CUDA function calls.
        val x = let("expf", floatInput(index).read)
        val y = let("exp", doubleInput(index).read)
        val floatOps: Vector[Expr[Float] => Expr[Float]] = Vector(exp[Float], log[Float], sqrt[Float], rsqrt[Float], tanh[Float])
        val doubleOps: Vector[Expr[Double] => Expr[Double]] = Vector(exp[Double], log[Double], sqrt[Double], rsqrt[Double], tanh[Double])
        floatOps.zipWithIndex.foreach { (operation, slot) =>
          floatOutput(literal(slot * count) + index) := operation(x)
        }
        doubleOps.zipWithIndex.foreach { (operation, slot) =>
          doubleOutput(literal(slot * count) + index) := operation(y)
        }
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "device_math.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val floatInput = context.allocate[Float](count).toOption.get
      val doubleInput = context.allocate[Double](count).toOption.get
      val floatOutput = context.allocate[Float](count * 5).toOption.get
      val doubleOutput = context.allocate[Double](count * 5).toOption.get
      assertEquals(floatInput.copyFrom(values.map(_.toFloat).toArray), Right(()))
      assertEquals(doubleInput.copyFrom(values.toArray), Right(()))
      assertEquals(function.launch(
        definition.bind((floatInput, doubleInput, floatOutput, doubleOutput)),
        LaunchConfig(Grid.x(1), LaunchBlock.x(128))
      ), Right(()))
      assertEquals(context.synchronize(), Right(()))
      verify(floatOutput.copyToArray().toOption.get.map(_.toDouble).toVector,
        values.map(_.toFloat.toDouble), tolerance = 2e-6, singlePrecision = true)
      verify(doubleOutput.copyToArray().toOption.get.toVector, values,
        tolerance = 1e-12, singlePrecision = false)
    finally context.close()

  private def verify(actual: Vector[Double], inputs: Vector[Double], tolerance: Double, singlePrecision: Boolean): Unit =
    val operations: Vector[Double => Double] = Vector(math.exp, math.log, math.sqrt, x => 1.0 / math.sqrt(x), math.tanh)
    val expected = operations.flatMap(operation => inputs.map(operation)).map { value =>
      if singlePrecision then value.toFloat.toDouble else value
    }
    assertEquals(actual.size, expected.size)
    actual.zip(expected).zipWithIndex.foreach { case ((result, reference), index) =>
      val clue = s"result $index: obtained $result, expected $reference"
      if reference.isNaN then assert(result.isNaN, clue)
      else if reference == 0.0 then
        assertEquals(java.lang.Double.doubleToRawLongBits(result), java.lang.Double.doubleToRawLongBits(reference), clue)
      else if java.lang.Double.isInfinite(reference) then assertEquals(result, reference, clue)
      else assert(math.abs(result - reference) <= tolerance * math.max(1.0, math.abs(reference)), clue)
    }
