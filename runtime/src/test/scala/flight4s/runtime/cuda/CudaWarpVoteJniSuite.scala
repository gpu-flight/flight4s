package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.UInt

class CudaWarpVoteJniSuite extends FunSuite:
  test("CUDA warp votes match exact masks and predicates for full partial and disjoint groups"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val definition = kernel("warpVotes", params((
      input[UInt]("masks"), input[Boolean]("predicates"), output[UInt]("ballots"),
      output[Boolean]("allResults"), output[Boolean]("anyResults"), value[Int]("count")
    ))) { bindings =>
      val (masks, predicates, ballots, allResults, anyResults, count) = bindings
      val linearThread = let("linearThread", threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
      val blockSize = blockDim.x * blockDim.y * blockDim.z
      val index = let("index", blockIdx.x * blockSize + linearThread)
      val mask = let("mask", masks(index).read)
      val predicate = let("predicate", predicates(index).read)
      val ballot = warp.ballot("ballot", mask, predicate)
      val all = warp.all("all", mask, predicate)
      val any = warp.any("any", mask, predicate)
      ballots(index) := ballot
      ballots(count + index) := ballot
      allResults(index) := all
      anyResults(index) := any
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "warp_votes.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      for block <- Vector(LaunchBlock.x(128), LaunchBlock.x(35), LaunchBlock.xyz(8, 4, 2)) do
        val blockSize = block.x * block.y * block.z
        val count = blockSize * 3
        val masks = context.allocate[UInt](count).toOption.get
        val predicates = context.allocate[Boolean](count).toOption.get
        val ballots = context.allocate[UInt](count * 2).toOption.get
        val allResults = context.allocate[Boolean](count).toOption.get
        val anyResults = context.allocate[Boolean](count).toOption.get
        try
          for partition <- 0 until 3; pattern <- 0 until 4 do
            val maskBits = Array.tabulate(count) { index =>
              val thread = index % blockSize
              val lane = thread % 32
              val lanes = math.min(32, blockSize - (thread / 32) * 32)
              val active = if lanes == 32 then -1 else (1 << lanes) - 1
              val group = partition match
                case 0 => -1
                case 1 => if lane < 16 then 0x0000ffff else 0xffff0000
                case 2 => if lane % 2 == 0 then 0x55555555 else 0xaaaaaaaa
              group & active
            }
            val values = Array.tabulate(count) { index => pattern match
              case 0 => false
              case 1 => true
              case 2 => (index % blockSize) % 32 == 0
              case 3 => index % 5 < 2
            }
            assertEquals(masks.copyFrom(maskBits.map(UInt.fromBits)), Right(()))
            assertEquals(predicates.copyFrom(values), Right(()))
            assertEquals(function.launch(definition.bind((masks, predicates, ballots, allResults, anyResults, count)),
              LaunchConfig(Grid.x(3), block)), Right(()))
            assertEquals(context.synchronize(), Right(()))
            val actualBallots = ballots.copyToArray().toOption.get.map(_.toIntBits)
            val actualAll = allResults.copyToArray().toOption.get
            val actualAny = anyResults.copyToArray().toOption.get
            for index <- 0 until count do
              val blockBase = (index / blockSize) * blockSize
              val warpBase = blockBase + ((index % blockSize) / 32) * 32
              val members = (0 until 32).filter(lane => ((maskBits(index) >>> lane) & 1) != 0)
              val expectedBallot = members.foldLeft(0) { (bits, lane) =>
                if values(warpBase + lane) then bits | (1 << lane) else bits
              }
              val clue = s"block=$block partition=$partition pattern=$pattern index=$index"
              assertEquals(actualBallots(index), expectedBallot, clue)
              assertEquals(actualBallots(count + index), expectedBallot, clue)
              assertEquals(actualAll(index), members.forall(lane => values(warpBase + lane)), clue)
              assertEquals(actualAny(index), members.exists(lane => values(warpBase + lane)), clue)
        finally
          masks.close()
          predicates.close()
          ballots.close()
          allResults.close()
          anyResults.close()
    finally context.close()
