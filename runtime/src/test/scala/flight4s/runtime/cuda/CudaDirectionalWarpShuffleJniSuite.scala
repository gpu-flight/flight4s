package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaDirectionalWarpShuffleJniSuite extends FunSuite:
  test("CUDA directional shuffles preserve bits and obey subgroup boundaries including XOR cross-group reads"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val integers = Array.tabulate(384)(i => Int.MinValue + i * 15485863)
      val floats = Array.tabulate(384) { i => java.lang.Float.intBitsToFloat(i % 8 match
        case 0 => 0
        case 1 => Int.MinValue
        case 2 => 0x7f800000
        case 3 => 0xff800000
        case 4 => 0x7fc00000 | i
        case 5 => 0xffc00000 | i
        case 6 => i
        case 7 => 0x3f800000 | i
      ) }
      val doubles = Array.tabulate(384) { i => java.lang.Double.longBitsToDouble(i % 8 match
        case 0 => 0L
        case 1 => Long.MinValue
        case 2 => 0x7ff0000000000000L
        case 3 => 0xfff0000000000000L
        case 4 => 0x7ff8000000000000L | i.toLong
        case 5 => 0xfff8000000000000L | i.toLong
        case 6 => i.toLong
        case 7 => 0x3ff0000000000000L | i.toLong
      ) }
      for width <- Vector(1, 2, 4, 8, 16, 32) do
        check(context, integers, (x: Int) => x.toLong, width)
        check(context, integers.map(UInt.fromBits), (x: UInt) => x.toIntBits.toLong, width)
        check(context, floats, (x: Float) => java.lang.Float.floatToRawIntBits(x).toLong, width)
        check(context, doubles, java.lang.Double.doubleToRawLongBits, width)
    finally context.close()

  private def sourceLane(direction: Int, lane: Int, selector: Int, width: Int): Int =
    val start = lane / width * width
    val candidate = direction match
      case 0 => lane - selector
      case 1 => lane + selector
      case 2 => lane ^ selector
    if (direction == 0 && candidate < start) || (direction != 0 && candidate >= start + width) then lane
    else candidate

  private def check[T](context: CudaContext, data: Array[T], bits: T => Long, width: Int)(using
      shuffleType: WarpShuffleType[T], codec: CudaHostCodec[T]
  ): Unit =
    val definition = kernel("directions", params(
      input[T]("source"), output[T]("received"), input[UInt]("masks"), input[UInt]("deltas"),
      input[Int]("laneMasks"), value[Int]("count")
    )) { bindings =>
      val (source, received, masks, deltas, laneMasks, count) = bindings
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      val up = warp.shuffleUp("up", masks(index).read, source(index).read, deltas(index).read, width)
      val down = warp.shuffleDown("down", masks(index).read, source(index).read, deltas(index).read, width)
      val xor = warp.shuffleXor("xorValue", masks(index).read, source(index).read, laneMasks(index).read, width)
      Vector(up, down, xor).zipWithIndex.foreach { (snapshot, direction) =>
        received(count * literal(direction) + index) := snapshot
        received(count * literal(direction + 3) + index) := snapshot
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(
      generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
    )
    val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "directional_shuffles.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).fold(failure => fail(failure.message), identity)
    try
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      for blockSize <- Vector(128, 35) do
        val count = blockSize * 3
        val values = data.take(count)
        val source = context.allocate[T](count).toOption.get
        val received = context.allocate[T](count * 6).toOption.get
        val masks = context.allocate[UInt](count).toOption.get
        val deltas = context.allocate[UInt](count).toOption.get
        val laneMasks = context.allocate[Int](count).toOption.get
        try
          assertEquals(source.copyFrom(values), Right(()))
          def active(i: Int): Int = math.min(32, blockSize - ((i % blockSize) / 32) * 32)
          val maskBits = Array.tabulate(count)(i => if active(i) == 32 then -1 else (1 << active(i)) - 1)
          assertEquals(masks.copyFrom(maskBits.map(UInt.fromBits)), Right(()))
          for pattern <- 0 until 6 do
            def proposed(i: Int): Int = pattern match
              case 0 => 0
              case 1 => 1
              case 2 => width - 1
              case 3 => math.min(width, 31)
              case 4 => 31
              case 5 => i * 13 % 32
            // An incomplete physical warp cannot supply missing source registers.
            val deltaValues = Array.tabulate(count) { i =>
              val lane = (i % blockSize) % 32
              val delta = proposed(i)
              if sourceLane(0, lane, delta, width) < active(i) && sourceLane(1, lane, delta, width) < active(i) then delta
              else 0
            }
            val xorValues = Array.tabulate(count) { i =>
              val lane = (i % blockSize) % 32
              if sourceLane(2, lane, proposed(i), width) < active(i) then proposed(i) else 0
            }
            assertEquals(deltas.copyFrom(deltaValues.map(UInt.fromBits)), Right(()))
            assertEquals(laneMasks.copyFrom(xorValues), Right(()))
            assertEquals(function.launch(definition.bind((source, received, masks, deltas, laneMasks, count)),
              LaunchConfig(Grid.x(3), LaunchBlock.x(blockSize))), Right(()))
            assertEquals(context.synchronize(), Right(()))
            val actual = received.copyToArray().toOption.get
            for i <- 0 until count; direction <- 0 until 3 do
              val lane = (i % blockSize) % 32
              val selector = if direction == 2 then xorValues(i) else deltaValues(i)
              val selected = i - lane + sourceLane(direction, lane, selector, width)
              val clue = s"${shuffleType.cudaName} direction=$direction width=$width block=$blockSize pattern=$pattern i=$i"
              assertEquals(bits(actual(direction * count + i)), bits(values(selected)), clue)
              assertEquals(bits(actual((direction + 3) * count + i)), bits(values(selected)), clue)
        finally
          source.close()
          received.close()
          masks.close()
          deltas.close()
          laneMasks.close()
    finally module.close()
