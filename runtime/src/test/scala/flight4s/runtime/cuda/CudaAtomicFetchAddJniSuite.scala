package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.UInt

class CudaAtomicFetchAddJniSuite extends FunSuite:
  test("atomic captures return unique old values and stable snapshots across global and shared CUDA updates"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val count = 4099
    val blockSize = 128
    val blocks = (count + blockSize - 1) / blockSize
    val definition = kernel("atomicTickets", params((
      inOut[Int]("intCounter"), inOut[UInt]("uintCounter"), inOut[Float]("floatCounter"),
      output[Int]("ints"), output[UInt]("uints"), output[Float]("floats")
    ))) { bindings =>
      val (intCounter, uintCounter, floatCounter, ints, uints, floats) = bindings
      val sharedInts = sharedArray[Int]("sharedInts", 1)
      val sharedUInts = sharedArray[UInt]("sharedUInts", 1)
      val sharedFloats = sharedArray[Float]("sharedFloats", 1)
      when(threadIdx.x === literal(0)) {
        sharedInts(literal(0)) := literal(11)
        sharedUInts(literal(0)) := literal(UInt.fromBits(100))
        sharedFloats(literal(0)) := literal(0.25f)
      }
      barrier()
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < literal(count)) {
        val globalInt = atomicFetchAdd("globalInt", intCounter(literal(0)), literal(1))
        val globalUInt = atomicFetchAdd("globalUInt", uintCounter(literal(0)), literal(UInt.fromBits(1)))
        val globalFloat = atomicFetchAdd("globalFloat", floatCounter(literal(0)), literal(0.5f))
        val sharedInt = atomicFetchAdd("sharedInt", sharedInts(literal(0)), literal(1))
        val sharedUInt = atomicFetchAdd("sharedUInt", sharedUInts(literal(0)), literal(UInt.fromBits(1)))
        val sharedFloat = atomicFetchAdd("sharedFloat", sharedFloats(literal(0)), literal(0.5f))
        ints(index) := globalInt
        uints(index) := globalUInt
        floats(index) := globalFloat
        ints(literal(count) + index) := sharedInt
        uints(literal(count) + index) := sharedUInt
        floats(literal(count) + index) := sharedFloat
        ints(literal(count * 2) + index) := globalInt
        uints(literal(count * 2) + index) := globalUInt
        floats(literal(count * 2) + index) := globalFloat
        ints(literal(count * 3) + index) := sharedInt
        uints(literal(count * 3) + index) := sharedUInt
        floats(literal(count * 3) + index) := sharedFloat
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "atomic_tickets.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val intCounter = context.allocate[Int](1).toOption.get
      val uintCounter = context.allocate[UInt](1).toOption.get
      val floatCounter = context.allocate[Float](1).toOption.get
      val ints = context.allocate[Int](count * 4).toOption.get
      val uints = context.allocate[UInt](count * 4).toOption.get
      val floats = context.allocate[Float](count * 4).toOption.get
      for run <- 0 until 5 do
        assertEquals(intCounter.copyFrom(Array(11)), Right(()))
        assertEquals(uintCounter.copyFrom(Array(UInt.fromBits(100))), Right(()))
        assertEquals(floatCounter.copyFrom(Array(0.25f)), Right(()))
        assertEquals(function.launch(definition.bind((intCounter, uintCounter, floatCounter, ints, uints, floats)),
          LaunchConfig(Grid.x(blocks), LaunchBlock.x(blockSize))), Right(()))
        assertEquals(context.synchronize(), Right(()))
        val i = ints.copyToArray().toOption.get.toVector
        val u = uints.copyToArray().toOption.get.map(_.toIntBits).toVector
        val f = floats.copyToArray().toOption.get.toVector
        assertEquals(i.take(count).sorted, Vector.tabulate(count)(_ + 11), s"Int global run $run")
        assertEquals(u.take(count).sorted, Vector.tabulate(count)(_ + 100), s"UInt global run $run")
        assertEquals(f.take(count).sorted, Vector.tabulate(count)(_ * 0.5f + 0.25f), s"Float global run $run")
        assertEquals(i.take(count * 2), i.drop(count * 2))
        assertEquals(u.take(count * 2), u.drop(count * 2))
        assertEquals(f.take(count * 2), f.drop(count * 2))
        for block <- 0 until blocks do
          val from = count + block * blockSize
          val until = count + math.min(count, (block + 1) * blockSize)
          assertEquals(i.slice(from, until).sorted, Vector.tabulate(until - from)(_ + 11))
          assertEquals(u.slice(from, until).sorted, Vector.tabulate(until - from)(_ + 100))
          assertEquals(f.slice(from, until).sorted, Vector.tabulate(until - from)(_ * 0.5f + 0.25f))
        assertEquals(intCounter.copyToArray().toOption.get.toVector, Vector(count + 11))
        assertEquals(uintCounter.copyToArray().toOption.get.map(_.toIntBits).toVector, Vector(count + 100))
        assertEquals(floatCounter.copyToArray().toOption.get.toVector, Vector(count * 0.5f + 0.25f))
    finally context.close()
