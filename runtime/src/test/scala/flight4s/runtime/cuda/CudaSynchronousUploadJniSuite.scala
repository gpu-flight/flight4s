package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaSynchronousUploadJniSuite extends FunSuite:
  test("pageable uploads are immediately consumable by nonblocking streams"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val definition = kernel("copyUploaded", params(input[Int]("source"), output[Int]("out"), value[Int]("n"))) { p =>
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < p._3) { p._2(index) := p._1(index).read }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated))
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "synchronous_upload.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      val function = module.function(generated).toOption.get
      val stream = context.createStream(CudaStreamMode.NonBlocking).toOption.get
      var checked = 0
      for count <- Vector(32, 4097, 32768, 65537, 262147) do
        val source = context.allocate[Int](count).toOption.get
        val out = context.allocate[Int](count + 32).toOption.get
        val arguments = definition.bind((source, out, count))
        val config = LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))
        for iteration <- 0 until 8 do
          val values = Array.tabulate(count)(i => (i * 1664525) ^ (iteration * 1013904223))
          assertEquals(out.copyFrom(Array.fill(count + 32)(-123)), Right(()))
          assertEquals(source.copyFrom(values), Right(()))
          // No context/default-stream completion call between upload and launch.
          assertEquals(function.launch(arguments, config, stream), Right(()))
          assertEquals(stream.synchronize(), Right(()))
          val actual = out.copyToArray().toOption.get
          assert(java.util.Arrays.equals(actual.take(count), values), s"count=$count iteration=$iteration")
          assertEquals(actual.takeRight(32).toVector, Vector.fill(32)(-123))
          checked += count
      assertEquals(checked, 2916648)
    finally context.close()
