package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaGraphJniSuite extends FunSuite:
  private def scaleKernel = kernel("graphScale", params(input[Float]("data"), output[Float]("out"),
      value[Int]("count"), value[Float]("factor"))) { p =>
    val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
    when(index < p._3) { p._2(index) := p._1(index).read * p._4 + literal(1.0f) }
  }

  test("captured gemm and kernel sequences replay deterministically over pooled buffers"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val definition = scaleKernel
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "graph_scale.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      val function = module.function(generated).toOption.get
      val blas = context.createBlas().toOption.get
      val pool = context.createPool(MemoryPoolConfig()).toOption.get
      val stream = context.createStream().toOption.get
      val side = 32
      val shape = GemmShape(side, side, side)
      val elements = side * side
      val factor = 2.0f
      val tolerance = 5e-4

      val a = pool.allocate[Float](elements).toOption.get
      val b = pool.allocate[Float](elements).toOption.get
      val c = pool.allocate[Float](elements).toOption.get
      val out = pool.allocate[Float](elements).toOption.get

      val graph = stream.capture {
        assertEquals(blas.gemmF32(a, b, c, shape, stream = Some(stream)), Right(()))
        assertEquals(function.launch(definition.bind((c, out, elements, factor)),
          LaunchConfig(Grid.x((elements + 127) / 128), LaunchBlock.x(128)), stream), Right(()))
      }.toOption.get

      def inputs(round: Int) =
        (Array.tabulate(elements)(i => ((i * 13 + round * 3) % 23 - 11) * 0.01f),
          Array.tabulate(elements)(i => ((i * 7 + round * 5) % 19 - 9) * 0.02f))
      def reference(aValues: Array[Float], bValues: Array[Float]) =
        Array.tabulate(elements) { index =>
          val row = index / side
          val column = index % side
          var accumulated = 0.0
          var inner = 0
          while inner < side do
            accumulated += aValues(row * side + inner).toDouble * bValues(inner * side + column).toDouble
            inner += 1
          accumulated * factor + 1.0
        }

      var firstReplay = Vector.empty[Float]
      for round <- 0 until 4 do
        // Rounds 0 and 1 share inputs to prove replay determinism bit for bit.
        val (aValues, bValues) = inputs(if round <= 1 then 0 else round)
        assertEquals(a.copyFrom(aValues), Right(()))
        assertEquals(b.copyFrom(bValues), Right(()))
        assertEquals(graph.launch(stream), Right(()))
        assertEquals(stream.synchronize(), Right(()))
        val produced = out.copyToArray().toOption.get
        val expected = reference(aValues, bValues)
        for index <- 0 until elements do
          assert(math.abs(produced(index) - expected(index)) <= tolerance,
            s"round $round [$index]: ${produced(index)} vs ${expected(index)}")
        if round == 0 then firstReplay = produced.toVector
        if round == 1 then assertEquals(produced.toVector, firstReplay)

      a.close(); b.close(); c.close(); out.close()
      assertEquals(pool.statistics.cachedBytes, 0L)
      assertEquals(graph.close(), Right(()))
      assert(pool.statistics.cachedBytes > 0L)
      val reused = pool.allocate[Float](elements).toOption.get
      assertEquals(pool.statistics.hits, 1L)
      reused.close()
      assertEquals(blas.close(), Right(()))
      assertEquals(pool.close(), Right(()))
    finally context.close()
