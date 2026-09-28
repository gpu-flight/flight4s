package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaMutableLoopBoundJniSuite extends FunSuite:
  test("CUDA loop rechecks an upper-bound local changed in its body"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val definition = kernel("changing_limit", params(output[Int]("out"))) { p =>
        scoped {
          val limit = local("limit", literal(8))
          val visits = local("visits", literal(0))
          gpuFor("i", literal(0), limit.read) { _ =>
            visits := visits.read + literal(1)
            limit := literal(0)
          }
          p._1(literal(0)) := visits.read
        }
        scoped {
          val limit = local("limit", literal(1))
          val visits = local("visits", literal(0))
          gpuFor("i", literal(0), limit.read) { _ =>
            visits := visits.read + literal(1)
            limit := literal(4)
          }
          p._1(literal(1)) := visits.read
        }
        scoped {
          val limit = local("limit", literal(8))
          val visits = local("visits", literal(0))
          gpuFor("i", literal(0), limit.read) { _ =>
            visits := visits.read + literal(1)
            accumulate(limit, literal(-1))
          }
          p._1(literal(2)) := visits.read
        }
        scoped {
          val limit = local("limit", literal(8))
          val visits = local("visits", literal(0))
          gpuFor("i", literal(0), limit.read) { i =>
            when(i === literal(0)) {
              scoped { gpuFor("j", literal(0), literal(1)) { _ => limit := literal(3) } }
            }
            visits := visits.read + literal(1)
          }
          p._1(literal(3)) := visits.read
        }
        scoped {
          val start = local("start", literal(2))
          val visits = local("visits", literal(0))
          gpuFor("i", start.read, literal(8)) { _ =>
            visits := visits.read + literal(1)
            start := literal(0)
          }
          p._1(literal(4)) := visits.read
        }
        scoped {
          val limit = local("limit", literal(8))
          val snapshot = let("snapshot", limit.read)
          val visits = local("visits", literal(0))
          gpuFor("i", literal(0), snapshot) { _ =>
            visits := visits.read + literal(1)
            limit := literal(0)
          }
          p._1(literal(5)) := visits.read
        }
        scoped {
          val limit = local("limit", literal(8))
          val extra = local("extra", literal(2))
          val visits = local("visits", literal(0))
          gpuFor("i", literal(0), limit.read + extra.read) { _ =>
            visits := visits.read + literal(1)
            limit := literal(0)
          }
          p._1(literal(6)) := visits.read
        }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "changing_limit.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      try
        val function = module.function(generated).toOption.get
        val out = context.allocate[Int](7).toOption.get
        try
          assertEquals(function.launch(definition.bind(Tuple1(out)), LaunchConfig(Grid.x(1), LaunchBlock.x(1))), Right(()))
          assertEquals(context.synchronize(), Right(()))
          assertEquals(out.copyToArray().toOption.get.toVector, Vector(1, 4, 4, 3, 6, 8, 2))
        finally out.close()
      finally module.close()
    finally context.close()
