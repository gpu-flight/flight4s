package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.unsafe.raw.RawCuda

class CudaRequiredBlockShapeJniSuite extends FunSuite:
  for rawLaunch <- Vector(false, true); explicit <- Vector(false, true) do
    test(s"CUDA rejects wrong shapes and permits a valid retry: raw=$rawLaunch explicit=$explicit"):
      assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
      val context = CudaContext.open(0) match
        case Right(context) => context
        case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
          assume(false, "CUDA device is not available")
          throw AssertionError("unreachable")
        case Left(failure) => fail(failure.message)
      try
        val required = LaunchBlock.xyz(4, 2, 2)
        val definition = kernel("requiredShape", params(output[Int]("out"))) { p =>
          val i = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
          p._1(i) := i
        }.requiringBlock(required)
        val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
        val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
        val raw = RawCuda.kernel(definition.name, definition.signature, generated.cudaSource,
          generated.compilerOptions, generated.launchRequirements)
        val compiled = if rawLaunch then NvrtcCompiler.compile(raw, context.computeCapability, "required_shape.cu")
          else NvrtcCompiler.compile(generatedModule, context.computeCapability, "required_shape.cu")
        val artifact = compiled.fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
        val module = context.load(artifact).fold(failure => fail(failure.message), identity)
        try
          val function = if rawLaunch then module.function(raw).toOption.get else module.function(generated).toOption.get
          val stream = if explicit then Some(context.createStream().toOption.get) else None
          val out = context.allocate[Int](16).toOption.get
          try
            assertEquals(out.copyFrom(Array.fill(16)(-1)), Right(()))
            def launch(shape: LaunchBlock) =
              val config = LaunchConfig(Grid.x(1), shape)
              stream match
                case Some(value) =>
                  if rawLaunch then function.launch(raw.bind(Tuple1(out)), config, value)
                  else function.launch(definition.bind(Tuple1(out)), config, value)
                case None =>
                  if rawLaunch then function.launch(raw.bind(Tuple1(out)), config)
                  else function.launch(definition.bind(Tuple1(out)), config)
            Vector(LaunchBlock.x(16), LaunchBlock.xyz(4, 4, 1), LaunchBlock.xyz(4, 2, 1)).foreach { wrong =>
              assertEquals(launch(wrong), Left(CudaLaunchFailure.BlockShapeMismatch(definition.name, required, wrong)))
            }
            assertEquals(context.synchronize(), Right(()))
            assertEquals(out.copyToArray().toOption.get.toVector, Vector.fill(16)(-1))
            assertEquals(launch(required), Right(()))
            assertEquals(context.synchronize(), Right(()))
            assertEquals(out.copyToArray().toOption.get.toVector, Vector.tabulate(16)(identity))
          finally out.close()
        finally module.close()
      finally context.close()
