package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaGridDimensionJniSuite extends FunSuite:
  test("one compiled kernel observes each launch grid on default and explicit streams"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val shape = LaunchBlock.xyz(4, 2, 2)
      val definition = kernel("gridDimensions", params(output[Int]("out"))) { p =>
        val lane = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
        when(lane === literal(0)) {
          val block = let("block", blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))
          val offset = block * literal(8)
          p._1(offset) := gridDim.x
          p._1(offset + literal(1)) := gridDim.y
          p._1(offset + literal(2)) := gridDim.z
          p._1(offset + literal(3)) := block
          p._1(offset + literal(4)) := gpuRange("i", literal(0), gridDim.y)
            .map(i => i + gridDim.z).sum(literal(0))
          p._1(offset + literal(5)) := choose(literal(-1) < gridDim.x)(literal(1))(literal(0))
          p._1(offset + literal(6)) := literal(-17) / (gridDim.x + literal(1))
          p._1(offset + literal(7)) := literal(-17) % (gridDim.z + literal(1))
        }
      }.requiringBlock(shape)
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      assert(!generated.cudaSource.contains("blockDim."))
      assert(generated.cudaSource.contains("gridDim.x"))
      val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "gridDimensions.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      try
        val function = module.function(generated).toOption.get
        val stream = context.createStream().toOption.get
        try
          val grids = Vector(Grid.x(1), Grid.x(7), Grid.xy(3, 5), Grid.xyz(3, 2, 4), Grid.xyz(2, 3, 5))
          var checkedValues = 0
          for grid <- grids; explicit <- Vector(false, true) do
            val blocks = grid.x * grid.y * grid.z
            val out = context.allocate[Int](blocks * 8).toOption.get
            try
              assertEquals(out.copyFrom(Array.fill(blocks * 8)(Int.MinValue)), Right(()))
              val invocation = definition.bind(Tuple1(out))
              val config = LaunchConfig(grid, shape)
              val submitted = if explicit then function.launch(invocation, config, stream)
                else function.launch(invocation, config)
              assertEquals(submitted, Right(()))
              assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
              val expected = Vector.tabulate(blocks) { block =>
                Vector(grid.x, grid.y, grid.z, block, grid.y * (grid.y - 1) / 2 + grid.y * grid.z,
                  1, -17 / (grid.x + 1), -17 % (grid.z + 1))
              }.flatten
              assertEquals(out.copyToArray().toOption.get.toVector, expected, s"grid=$grid explicit=$explicit")
              checkedValues += expected.size
            finally out.close()
          assertEquals(checkedValues, 1232)
        finally stream.close()
      finally module.close()
    finally context.close()
