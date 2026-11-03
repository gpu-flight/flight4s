package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaBlasJniSuite extends FunSuite:
  // The second oracle: an explicit-DSL serial-k product sum guarding against
  // systematic layout mistakes that a CPU-only comparison could share. The
  // alpha/beta algebra stays on the cuBLASLt side, checked against the CPU.
  private def naiveGemm = kernel("naiveGemm", params(input[Float]("a"), input[Float]("b"),
      output[Float]("c"), value[Int]("m"), value[Int]("n"), value[Int]("k"))) { p =>
    val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
    when(index < p._4 * p._5) {
      val row = let("row", index / p._5)
      val column = let("column", index % p._5)
      val total = local("total", literal(0.0f))
      gpuFor("inner", literal(0), p._6) { inner =>
        total := total.read +
          p._1(row * p._6 + inner).read * p._2(inner * p._5 + column).read
      }
      p._3(index) := total.read
    }
  }

  test("cuBLASLt GEMM matches the CPU reference and the naive kernel oracle"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val definition = naiveGemm
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "naive_gemm.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      val function = module.function(generated).toOption.get
      val blas = context.createBlas().toOption.get
      val pool = context.createPool(MemoryPoolConfig()).toOption.get
      val stream = context.createStream().toOption.get
      val tolerance = 5e-4

      val cases = Vector(
        (GemmShape(1, 1, 1), 1.0f, 0.0f, false, false),
        (GemmShape(1, 5, 1), 1.0f, 0.0f, false, false),
        (GemmShape(3, 1, 2), 1.0f, 0.0f, false, true),
        (GemmShape(33, 17, 65), 1.0f, 0.0f, false, false),
        (GemmShape(64, 64, 64), 0.5f, 0.0f, true, false),
        (GemmShape(64, 64, 64), 1.0f, 1.0f, true, true),
        (GemmShape(128, 96, 192), 1.0f, 0.0f, true, false)
      )
      for (shape, alpha, beta, useWorkspace, useStream) <- cases do
        val label = s"${shape.m}x${shape.n}x${shape.k} alpha=$alpha beta=$beta"
        val aValues = Array.tabulate(shape.m * shape.k)(i => ((i * 13) % 23 - 11) * 0.01f)
        val bValues = Array.tabulate(shape.k * shape.n)(i => ((i * 7) % 19 - 9) * 0.02f)
        val cSeed = Array.tabulate(shape.m * shape.n)(i => ((i * 5) % 17 - 8) * 0.05f)
        val rawReference = Array.tabulate(shape.m * shape.n) { index =>
          val row = index / shape.n
          val column = index % shape.n
          var accumulated = 0.0
          var inner = 0
          while inner < shape.k do
            accumulated += aValues(row * shape.k + inner).toDouble *
              bValues(inner * shape.n + column).toDouble
            inner += 1
          accumulated
        }
        val reference = Array.tabulate(shape.m * shape.n) { index =>
          alpha * rawReference(index) + beta * cSeed(index).toDouble
        }
        val a = context.allocate[Float](aValues.length).toOption.get
        val b = context.allocate[Float](bValues.length).toOption.get
        val cBlas = pool.allocate[Float](cSeed.length).toOption.get
        val cNaive = pool.allocate[Float](cSeed.length).toOption.get
        val workspace = if useWorkspace then Some(pool.allocate[Float](1024).toOption.get) else None
        try
          assertEquals(a.copyFrom(aValues), Right(()))
          assertEquals(b.copyFrom(bValues), Right(()))
          assertEquals(cBlas.copyFrom(cSeed), Right(()))
          assertEquals(cNaive.copyFrom(cSeed), Right(()))

          val streamChoice = if useStream then Some(stream) else None
          assertEquals(blas.gemmF32(a, b, cBlas, shape, alpha, beta, workspace, streamChoice), Right(()))
          val config = LaunchConfig(Grid.x((shape.m * shape.n + 127) / 128), LaunchBlock.x(128))
          assertEquals(function.launch(
            definition.bind((a, b, cNaive, shape.m, shape.n, shape.k)), config), Right(()))
          if useStream then assertEquals(stream.synchronize(), Right(()))
          assertEquals(context.synchronize(), Right(()))

          val producedBlas = cBlas.copyToArray().toOption.get
          val producedNaive = cNaive.copyToArray().toOption.get
          for index <- reference.indices do
            assert(math.abs(producedBlas(index) - reference(index)) <= tolerance,
              s"$label blas[$index]: ${producedBlas(index)} vs ${reference(index)}")
            assert(math.abs(producedNaive(index) - rawReference(index)) <= tolerance,
              s"$label naive[$index]: ${producedNaive(index)} vs ${rawReference(index)}")
          assertEquals(a.copyToArray().toOption.get.toVector, aValues.toVector, label)
          assertEquals(b.copyToArray().toOption.get.toVector, bValues.toVector, label)
        finally
          workspace.foreach(_.close())
          cNaive.close()
          cBlas.close()
          b.close()
          a.close()
      assertEquals(blas.close(), Right(()))
      assertEquals(pool.close(), Right(()))
    finally context.close()
