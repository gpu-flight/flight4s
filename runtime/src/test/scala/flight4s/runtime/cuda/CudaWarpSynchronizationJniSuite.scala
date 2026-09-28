package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.UInt

class CudaWarpSynchronizationJniSuite extends FunSuite:
  test("warp sync orders repeated shared and global exchanges across full partial and disjoint groups"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val iterations = 32
      val definition = kernel("warpExchange", params((
        inOut[Int]("globalScratch"), output[Int]("sharedResults"), output[Int]("globalResults"),
        input[UInt]("masks"), input[Int]("neighbors"), value[Int]("count"),
        value[Int]("pattern"), value[Int]("seed")
      ))) { bindings =>
        val (globalScratch, sharedResults, globalResults, masks, neighbors, count, pattern, seed) = bindings
        val tile = sharedArray[Int]("tile", 128)
        val thread = let("thread", threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
        val blockSize = let("blockSize", blockDim.x * blockDim.y * blockDim.z)
        val index = let("index", blockIdx.x * blockSize + thread)
        val neighbor = let("neighbor", neighbors(index).read)
        val mask = let("mask", masks(index).read)
        def exchange()(using BlockBuilder): Unit =
          gpuRange("iteration", literal(0), literal(iterations)).foreach { iteration =>
            val contribution = let("contribution", index * literal(17) + iteration * literal(100000) + seed)
            tile(thread) := contribution
            globalScratch(index) := contribution
            warp.sync(mask)
            val fromShared = let("fromShared", tile(neighbor).read)
            val fromGlobal = let("fromGlobal", globalScratch(blockIdx.x * blockSize + neighbor).read)
            // Finish all participant reads before any lane reuses its scratch slot.
            warp.sync(mask)
            sharedResults(iteration * count + index) := fromShared
            globalResults(iteration * count + index) := fromGlobal
          }
        gpuIf(pattern === literal(2)) {
          gpuIf((thread % literal(2)) === literal(0)) { exchange() } { exchange() }
        } { exchange() }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      assertEquals(generated.cudaSource.sliding("::__syncwarp".length).count(_ == "::__syncwarp"), 6)
      assert(!generated.cudaSource.contains("__syncthreads"))
      val generatedModule = GeneratedCudaModule(
        generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
      )
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "warp_sync.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      try
        val function = module.function(generated).fold(failure => fail(failure.message), identity)
        for block <- Vector(LaunchBlock.x(128), LaunchBlock.x(35), LaunchBlock.xyz(8, 4, 2)) do
          val blockSize = block.x * block.y * block.z
          val count = blockSize * 3
          val globalScratch = context.allocate[Int](count).toOption.get
          val sharedResults = context.allocate[Int](count * iterations).toOption.get
          val globalResults = context.allocate[Int](count * iterations).toOption.get
          val masks = context.allocate[UInt](count).toOption.get
          val neighbors = context.allocate[Int](count).toOption.get
          try
            for pattern <- 0 until 3 do
              val groups = Array.tabulate(count) { i =>
                val thread = i % blockSize
                val lane = thread % 32
                val active = math.min(32, blockSize - (thread / 32) * 32)
                (0 until active).filter { candidate => pattern match
                  case 0 => true
                  case 1 => candidate / 16 == lane / 16
                  case 2 => candidate % 2 == lane % 2
                }.toVector
              }
              val maskValues = groups.map(group => UInt.fromBits(group.foldLeft(0)((mask, lane) => mask | (1 << lane))))
              val neighborValues = Array.tabulate(count) { i =>
                val thread = i % blockSize
                val group = groups(i)
                (thread / 32) * 32 + group((group.indexOf(thread % 32) + 1) % group.size)
              }
              assertEquals(masks.copyFrom(maskValues), Right(()))
              assertEquals(neighbors.copyFrom(neighborValues), Right(()))
              for repeat <- 0 until 5 do
                val seed = repeat * 29
                assertEquals(function.launch(definition.bind((
                  globalScratch, sharedResults, globalResults, masks, neighbors, count, pattern, seed
                )), LaunchConfig(Grid.x(3), block)), Right(()))
                assertEquals(context.synchronize(), Right(()))
                val shared = sharedResults.copyToArray().toOption.get
                val global = globalResults.copyToArray().toOption.get
                for iteration <- 0 until iterations; i <- 0 until count do
                  val peer = (i / blockSize) * blockSize + neighborValues(i)
                  val expected = peer * 17 + iteration * 100000 + seed
                  val clue = s"block=$block pattern=$pattern repeat=$repeat iteration=$iteration i=$i"
                  assertEquals(shared(iteration * count + i), expected, clue)
                  assertEquals(global(iteration * count + i), expected, clue)
          finally
            globalScratch.close()
            sharedResults.close()
            globalResults.close()
            masks.close()
            neighbors.close()
      finally module.close()
    finally context.close()
