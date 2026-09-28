package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.UInt

class CudaAtomicAddJniSuite extends FunSuite:
  test("global and shared atomics preserve all Int UInt and Float updates under CUDA contention"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val count = 4099
    val blockSize = 128
    val blocks = (count + blockSize - 1) / blockSize
    val buckets = 7
    val slots = buckets + blocks
    val definition = kernel("atomicCounts", params(
      inOut[Int]("ints"), inOut[UInt]("uints"), inOut[Float]("floats")
    )) { bindings =>
      val (ints, uints, floats) = bindings
      val sharedInts = sharedArray[Int]("sharedInts", 1)
      val sharedUInts = sharedArray[UInt]("sharedUInts", 1)
      val sharedFloats = sharedArray[Float]("sharedFloats", 1)
      when(threadIdx.x === literal(0)) {
        sharedInts(literal(0)) := literal(0)
        sharedUInts(literal(0)) := literal(UInt.fromBits(0))
        sharedFloats(literal(0)) := literal(0.0f)
      }
      barrier()
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < literal(count)) {
        gpuRange("repeat", literal(0), literal(3)).foreach { _ =>
          atomicAdd(ints(index % literal(buckets)), literal(1))
          atomicAdd(uints(index % literal(buckets)), literal(UInt.fromBits(1)))
          atomicAdd(floats(index % literal(buckets)), literal(0.5f))
          atomicAdd(sharedInts(literal(0)), literal(1))
          atomicAdd(sharedUInts(literal(0)), literal(UInt.fromBits(1)))
          atomicAdd(sharedFloats(literal(0)), literal(0.5f))
        }
      }
      barrier()
      when(threadIdx.x === literal(0)) {
        ints(literal(buckets) + blockIdx.x) := sharedInts(literal(0)).read
        uints(literal(buckets) + blockIdx.x) := sharedUInts(literal(0)).read
        floats(literal(buckets) + blockIdx.x) := sharedFloats(literal(0)).read
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "atomic_counts.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val ints = context.allocate[Int](slots).toOption.get
      val uints = context.allocate[UInt](slots).toOption.get
      val floats = context.allocate[Float](slots).toOption.get
      val expected = Array.tabulate(slots) { slot =>
        if slot < buckets then (0 until count).count(_ % buckets == slot) * 3
        else math.min(blockSize, count - (slot - buckets) * blockSize) * 3
      }
      for repetition <- 0 until 5 do
        assertEquals(ints.copyFrom(Array.fill(slots)(0)), Right(()))
        assertEquals(uints.copyFrom(Array.fill(slots)(UInt.fromBits(0))), Right(()))
        assertEquals(floats.copyFrom(Array.fill(slots)(0.0f)), Right(()))
        assertEquals(function.launch(definition.bind((ints, uints, floats)),
          LaunchConfig(Grid.x(blocks), LaunchBlock.x(blockSize))), Right(()))
        assertEquals(context.synchronize(), Right(()))
        assertEquals(ints.copyToArray().toOption.get.toVector, expected.toVector, s"Int run $repetition")
        assertEquals(uints.copyToArray().toOption.get.map(_.toIntBits).toVector, expected.toVector, s"UInt run $repetition")
        assertEquals(floats.copyToArray().toOption.get.toVector, expected.map(_ * 0.5f).toVector, s"Float run $repetition")
    finally context.close()
