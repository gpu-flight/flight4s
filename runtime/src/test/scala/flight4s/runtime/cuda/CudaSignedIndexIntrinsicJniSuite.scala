package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaSignedIndexIntrinsicJniSuite extends FunSuite:
  test("CUDA indexing expressions preserve signed comparisons division and remainder"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      val definition = kernel("signedIndexes", params(output[Int]("out"))) { p =>
        val lane = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
        val block = blockIdx.x + literal(3) * (blockIdx.y + literal(2) * blockIdx.z)
        val offset = (block * literal(16) + lane) * literal(36)
        val indexes = Vector(threadIdx.x, threadIdx.y, threadIdx.z,
          blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z)
        indexes.zipWithIndex.foreach { (index, position) =>
          val out = offset + literal(position * 4)
          p._1(out) := choose(literal(-1) < index)(literal(1))(literal(0))
          p._1(out + literal(1)) := choose((index - literal(32)) < literal(0))(literal(1))(literal(0))
          p._1(out + literal(2)) := literal(-17) / (index + literal(1))
          p._1(out + literal(3)) := literal(-17) % (index + literal(1))
        }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "signedIndexes.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      try
        val out = context.allocate[Int](6912).toOption.get
        try
          assertEquals(out.copyFrom(Array.fill(6912)(Int.MinValue)), Right(()))
          assertEquals(module.function(generated).toOption.get.launch(definition.bind(Tuple1(out)),
            LaunchConfig(Grid.xyz(3, 2, 2), LaunchBlock.xyz(4, 2, 2))), Right(()))
          assertEquals(context.synchronize(), Right(()))
          val expected = (for
            bz <- 0 until 2; by <- 0 until 2; bx <- 0 until 3
            tz <- 0 until 2; ty <- 0 until 2; tx <- 0 until 4
            index <- Vector(tx, ty, tz, bx, by, bz, 4, 2, 2)
            result <- Vector(1, 1, -17 / (index + 1), -17 % (index + 1))
          yield result).toVector
          assertEquals(expected.size, 6912)
          assertEquals(out.copyToArray().toOption.get.toVector, expected)
        finally out.close()
      finally module.close()
    finally context.close()
