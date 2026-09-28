package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaBlockDimensionSpecializationJniSuite extends FunSuite:
  test("negative arithmetic has the same signed semantics before and after specialization"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val base = kernel("negativeGeometry", params(output[Int]("out"))) { p =>
        when(threadIdx.x === literal(0)) {
          p._1(literal(0)) := choose(literal(-1) < blockDim.x)(literal(1))(literal(0))
          p._1(literal(1)) := literal(-7) / blockDim.x
          p._1(literal(2)) := literal(-7) % blockDim.x
        }
      }
      val actual = Vector(false, true).map { specialized =>
        val definition = if specialized then base.requiringBlock(LaunchBlock.x(8)) else base
        val generated = CudaCodegen.generate(definition).toOption.get
        val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
        val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "negativeGeometry.cu")
          .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
        val module = context.load(artifact).toOption.get
        try
          val out = context.allocate[Int](3).toOption.get
          try
            assertEquals(module.function(generated).toOption.get.launch(definition.bind(Tuple1(out)),
              LaunchConfig(Grid.x(1), LaunchBlock.x(8))), Right(()))
            assertEquals(context.synchronize(), Right(()))
            out.copyToArray().toOption.get.toVector
          finally out.close()
        finally module.close()
      }
      assertEquals(actual, Vector.fill(2)(Vector(1, 0, -7)))
    finally context.close()

  test("specialized and dynamic geometry match CPU results across 1D, 2D, and 3D blocks"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val base = kernel("geometry", params(output[Int]("out"), value[Int]("seed"))) { p =>
        val lane = let("lane", threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
        val size = blockDim.x * blockDim.y * blockDim.z
        val offset = (blockIdx.x * size + lane) * literal(6)
        val scratch = sharedArray[Int]("scratch", 128)
        scratch(lane) := lane + p._2
        barrier()

        val bound = local("bound", blockDim.x + literal(2))
        val visits = local("visits", literal(0))
        gpuFor("visit", literal(0), bound.read) { _ =>
          visits := visits.read + literal(1)
          bound := blockDim.y
        }
        val sum = gpuRange("sumIndex", literal(0), blockDim.z)
          .map(i => lane + i + blockDim.x).sum(literal(0))
        val folded = gpuRange("foldIndex", literal(0), blockDim.x).by(3)
          .filter(i => (i % literal(2)) === literal(0))
          .map(i => i + blockDim.y)
          .foldLeft("total", p._2)((acc, element) => acc + element)
        val branch = local("branch", literal(1))
        when((lane % literal(2)) === literal(0)) { branch := blockDim.y }
        val selected = choose(blockDim.y === literal(1))(literal(111))(literal(222))
        p._1(offset) := size
        p._1(offset + literal(1)) := sum
        p._1(offset + literal(2)) := folded
        p._1(offset + literal(3)) := visits.read
        p._1(offset + literal(4)) := branch.read
        p._1(offset + literal(5)) := scratch((lane + literal(1)) % size).read + selected
      }
      val shapes = Vector(LaunchBlock.x(1), LaunchBlock.x(2), LaunchBlock.x(32), LaunchBlock.x(35),
        LaunchBlock.xy(8, 4), LaunchBlock.xyz(8, 4, 2), LaunchBlock.xyz(3, 5, 2))
      var checkedValues = 0
      for shape <- shapes; specialized <- Vector(false, true) do
        val definition = if specialized then base.requiringBlock(shape) else base
        val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
        assertEquals(generated.cudaSource.contains("blockDim."), !specialized)
        val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
        val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "geometry.cu")
          .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
        val module = context.load(artifact).fold(failure => fail(failure.message), identity)
        try
          val function = module.function(generated).toOption.get
          val count = shape.x * shape.y * shape.z
          val out = context.allocate[Int](count * 3 * 6).toOption.get
          try
            Vector(-7, 0, 31).foreach { seed =>
              assertEquals(out.copyFrom(Array.fill(count * 3 * 6)(Int.MinValue)), Right(()))
              assertEquals(function.launch(definition.bind((out, seed)), LaunchConfig(Grid.x(3), shape)), Right(()))
              assertEquals(context.synchronize(), Right(()))
              val expected = Vector.tabulate(count * 3) { global =>
                val lane = global % count
                Vector(count, shape.z * (lane + shape.x) + shape.z * (shape.z - 1) / 2,
                  seed + (0 until shape.x by 3).filter(_ % 2 == 0).map(_ + shape.y).sum,
                  shape.y, if lane % 2 == 0 then shape.y else 1,
                  (lane + 1) % count + seed + (if shape.y == 1 then 111 else 222))
              }.flatten
              assertEquals(out.copyToArray().toOption.get.toVector, expected,
                s"shape=$shape specialized=$specialized seed=$seed")
              checkedValues += expected.size
            }
          finally out.close()
        finally module.close()
      assertEquals(checkedValues, 21168)
    finally context.close()
