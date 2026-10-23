package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaMemoryPoolJniSuite extends FunSuite:
  test("pooled device and pinned buffers drive real launches and reuse blocks across rounds"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val definition = kernel("scalePooled", params(input[Float]("source"), output[Float]("out"),
        value[Int]("n"), value[Float]("factor"))) { p =>
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < p._3) { p._2(index) := p._1(index).read * p._4 }
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
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "memory_pool.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      val function = module.function(generated).toOption.get
      val pool = context.createPool(MemoryPoolConfig()).toOption.get
      val count = 4096
      val config = LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))
      for round <- 0 until 4 do
        val values = Array.tabulate(count)(i => ((i * 31 + round * 7) % 19 - 9) * 0.25f)
        val produced = pool.withPinned[Float, Vector[Float]](count) { staging =>
          pool.withBuffer[Float, Vector[Float]](count) { source =>
            pool.withBuffer[Float, Vector[Float]](count + 32) { out =>
              staging.copyFrom(values)
              for
                _ <- source.copyFrom(staging)
                _ <- out.copyFrom(Array.fill(count + 32)(-123.0f))
                _ = assertEquals(function.launch(definition.bind((source, out, count, 1.5f)), config), Right(()))
                _ <- context.synchronize()
                result <- out.copyToArray()
              yield result.toVector
            }
          }
        }.fold(failure => fail(s"round $round: ${failure.message}"), identity)
        assertEquals(produced.take(count), values.toVector.map(_ * 1.5f), s"round $round")
        assertEquals(produced.takeRight(32), Vector.fill(32)(-123.0f), s"round $round")
      assertEquals(pool.statistics.misses, 3L)
      assertEquals(pool.statistics.hits, 9L)
      assertEquals(pool.statistics.liveBytes, 0L)
      assertEquals(pool.trim(), Right(()))
      assertEquals(pool.statistics.cachedBytes, 0L)
      assertEquals(pool.close(), Right(()))
    finally context.close()
