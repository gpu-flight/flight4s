package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaBooleanJniSuite extends FunSuite:
  test("staged logical operators execute truth tables and guarded memory reads on CUDA"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val rows = 259
    val leftValues = Array.tabulate(rows)(i => i % 2 == 0)
    val rightValues = Array.tabulate(rows)(i => i % 4 < 2)
    val definition = kernel("logicalPredicates", params(
      input[Boolean]("left"), input[Boolean]("right"), input[Int]("source"), output[Boolean]("out")
    )) { bindings =>
      val (left, right, source, out) = bindings
      val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
      when(row < literal(rows)) {
        val a = left(row).read
        val b = right(row).read
        out(row) := a && b
        out(literal(rows) + row) := a || b
        out(literal(rows * 2) + row) := !a
        val index = let("index", row % literal(3) - literal(1))
        val inBounds = index >= literal(0) && index < literal(1)
        out(literal(rows * 3) + row) := inBounds && source(index).read > literal(0)
        out(literal(rows * 4) + row) := !inBounds || source(index).read > literal(0)
        out(literal(rows * 5) + row) := (index !== literal(0)) && literal(12) / index > literal(2)
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "logical_predicates.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val left = context.allocate[Boolean](rows).toOption.get
      val right = context.allocate[Boolean](rows).toOption.get
      val source = context.allocate[Int](1).toOption.get
      val out = context.allocate[Boolean](rows * 6).toOption.get
      assertEquals(left.copyFrom(leftValues), Right(()))
      assertEquals(right.copyFrom(rightValues), Right(()))
      for sourceValue <- Vector(-7, 7) do
        assertEquals(source.copyFrom(Array(sourceValue)), Right(()))
        assertEquals(function.launch(
          definition.bind((left, right, source, out)), LaunchConfig(Grid.x(3), LaunchBlock.x(128))
        ), Right(()))
        assertEquals(context.synchronize(), Right(()))
        val results = out.copyToArray().toOption.get
        for row <- 0 until rows do
          val index = row % 3 - 1
          val inBounds = index >= 0 && index < 1
          val expected = Vector(
            leftValues(row) && rightValues(row), leftValues(row) || rightValues(row), !leftValues(row),
            inBounds && sourceValue > 0, !inBounds || sourceValue > 0, index != 0 && 12 / index > 2
          )
          expected.zipWithIndex.foreach { (value, slot) =>
            assertEquals(results(slot * rows + row), value, s"source $sourceValue, row $row, slot $slot")
          }
    finally context.close()
