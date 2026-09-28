package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaWarpShuffleJniSuite extends FunSuite:
  test("CUDA direct shuffles preserve scalar bits across subgroup widths and partial warps"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val integerBits = Array.tabulate(384)(i => if i == 0 then Int.MinValue else i * 15485863)
      val floatBits = Array.tabulate(384) { i => i % 8 match
        case 0 => 0
        case 1 => Int.MinValue
        case 2 => 0x7f800000
        case 3 => 0xff800000
        case 4 => 0x7fc00000 | i
        case 5 => 0xffc00000 | i
        case 6 => i
        case 7 => 0x3f800000 | i
      }
      val doubleBits = Array.tabulate(384) { i => i % 8 match
        case 0 => 0L
        case 1 => Long.MinValue
        case 2 => 0x7ff0000000000000L
        case 3 => 0xfff0000000000000L
        case 4 => 0x7ff8000000000000L | i.toLong
        case 5 => 0xfff8000000000000L | i.toLong
        case 6 => i.toLong
        case 7 => 0x3ff0000000000000L | i.toLong
      }
      for width <- Vector(1, 2, 4, 8, 16, 32) do
        check(context, integerBits, (value: Int) => value.toLong, width)
        check(context, integerBits.map(UInt.fromBits), (value: UInt) => value.toIntBits.toLong, width)
        check(context, floatBits.map(java.lang.Float.intBitsToFloat),
          (value: Float) => java.lang.Float.floatToRawIntBits(value).toLong, width)
        check(context, doubleBits.map(java.lang.Double.longBitsToDouble), java.lang.Double.doubleToRawLongBits, width)
    finally context.close()

  private def check[T](context: CudaContext, data: Array[T], bits: T => Long, width: Int)(using
      shuffleType: WarpShuffleType[T], codec: CudaHostCodec[T]
  ): Unit =
    val definition = kernel("shuffleValues", params(
      input[T]("source"), output[T]("received"), input[UInt]("masks"), input[Int]("lanes"), value[Int]("count")
    )) { bindings =>
      val (source, received, masks, lanes, count) = bindings
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      val shuffled = warp.shuffle("shuffled", masks(index).read, source(index).read, lanes(index).read, width)
      received(index) := shuffled
      received(count + index) := shuffled
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(
      generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
    )
    val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "warp_shuffle.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).fold(failure => fail(failure.message), identity)
    try
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      for blockSize <- Vector(128, 35) do
        val count = blockSize * 3
        val values = data.take(count)
        val source = context.allocate[T](count).toOption.get
        val received = context.allocate[T](count * 2).toOption.get
        val masks = context.allocate[UInt](count).toOption.get
        val lanes = context.allocate[Int](count).toOption.get
        try
          assertEquals(source.copyFrom(values), Right(()))
          for pattern <- 0 until 2 do
            val subgroupStarts = Array.tabulate(count)(i => ((i % blockSize) % 32 / width) * width)
            val subgroupSizes = Array.tabulate(count) { i =>
              val thread = i % blockSize
              math.min(width, math.min(32, blockSize - (thread / 32) * 32) - subgroupStarts(i))
            }
            val maskBits = Array.tabulate(count) { i =>
              val size = subgroupSizes(i)
              (if size == 32 then -1 else (1 << size) - 1) << subgroupStarts(i)
            }
            val selectors = Array.tabulate(count) { i =>
              if pattern == 0 then 0
              else ((i % blockSize) + 1) % subgroupSizes(i) + width * 3
            }
            assertEquals(masks.copyFrom(maskBits.map(UInt.fromBits)), Right(()))
            assertEquals(lanes.copyFrom(selectors), Right(()))
            assertEquals(function.launch(definition.bind((source, received, masks, lanes, count)),
              LaunchConfig(Grid.x(3), LaunchBlock.x(blockSize))), Right(()))
            assertEquals(context.synchronize(), Right(()))
            val actual = received.copyToArray().toOption.get
            for i <- 0 until count do
              val warpStart = (i / blockSize) * blockSize + ((i % blockSize) / 32) * 32
              val selected = warpStart + subgroupStarts(i) + selectors(i) % width
              val clue = s"${shuffleType.cudaName} width=$width block=$blockSize pattern=$pattern i=$i"
              assertEquals(bits(actual(i)), bits(values(selected)), clue)
              assertEquals(bits(actual(count + i)), bits(values(selected)), clue)
        finally
          source.close()
          received.close()
          masks.close()
          lanes.close()
    finally module.close()
