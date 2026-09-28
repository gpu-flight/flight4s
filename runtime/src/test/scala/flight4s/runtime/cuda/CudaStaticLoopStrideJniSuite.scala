package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaStaticLoopStrideJniSuite extends FunSuite:
  private def withContext(body: CudaContext => Unit): Unit =
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try body(context)
    finally context.close()

  test("static strides match a wide host reference including Int endpoints and partial blocks"):
    withContext { context =>
      val slots = 16
      val sentinel = 0x5a5a5a5a
      Vector(1, 2, 3, 16, 31, 32, Int.MaxValue).foreach { step =>
        val random = new scala.util.Random(step.toLong)
        val cases = Vector(
          (0, 0), (8, 0), (-8, 8), (0, 9),
          (Int.MaxValue - 2, Int.MaxValue), (Int.MaxValue - 1, Int.MaxValue),
          (Int.MinValue, Int.MinValue + 3), (Int.MaxValue, Int.MaxValue)
        ) ++ (if step == Int.MaxValue then Vector((Int.MinValue, Int.MaxValue)) else Vector.empty) ++
          Vector.fill(257) {
            val from = random.nextInt()
            val distance = step.toLong * random.nextInt(7) + random.nextInt(2)
            (from, math.min(Int.MaxValue.toLong, from.toLong + distance).toInt)
          }
        val expected = cases.map { (from, until) =>
          Iterator.iterate(from.toLong)(_ + step.toLong).takeWhile(_ < until.toLong).map(_.toInt).toVector
        }
        assert(expected.forall(_.size <= slots))
        val definition = kernel("strided", params(input[Int]("starts"), input[Int]("ends"),
          output[Int]("values"), output[Int]("counts"), value[Int]("rows"))) { p =>
          val (starts, ends, values, counts, rows) = p
          val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
          when(row < rows) {
            val count = local("count", literal(0))
            gpuFor("i", starts(row).read, ends(row).read, step) { i =>
              values(row * literal(slots) + count.read) := i
              count := count.read + literal(1)
            }
            counts(row) := count.read
          }
        }
        val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
        val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
        val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "strided.cu")
          .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
        val module = context.load(artifact).fold(failure => fail(failure.message), identity)
        try
          val function = module.function(generated).toOption.get
          val starts = context.allocate[Int](cases.size).toOption.get
          val ends = context.allocate[Int](cases.size).toOption.get
          val values = context.allocate[Int](cases.size * slots).toOption.get
          val counts = context.allocate[Int](cases.size).toOption.get
          try
            assertEquals(starts.copyFrom(cases.map(_._1).toArray), Right(()))
            assertEquals(ends.copyFrom(cases.map(_._2).toArray), Right(()))
            assertEquals(values.copyFrom(Array.fill(cases.size * slots)(sentinel)), Right(()))
            assertEquals(function.launch(definition.bind((starts, ends, values, counts, cases.size)),
              LaunchConfig(Grid.x((cases.size + 127) / 128), LaunchBlock.x(128))), Right(()))
            assertEquals(context.synchronize(), Right(()))
            assertEquals(counts.copyToArray().toOption.get.toVector, expected.map(_.size), s"step=$step")
            assertEquals(values.copyToArray().toOption.get.toVector,
              expected.flatMap(_.padTo(slots, sentinel)), s"step=$step")
          finally
            counts.close()
            values.close()
            ends.close()
            starts.close()
        finally module.close()
      }
    }

  test("strided loops recheck changing bounds and preserve nested index values"):
    withContext { context =>
      val definition = kernel("changing", params(output[Int]("out"))) { p =>
        scoped {
          val limit = local("limit", literal(8))
          val count = local("count", literal(0))
          gpuFor("i", literal(0), limit.read, 3) { _ =>
            count := count.read + literal(1)
            limit := literal(0)
          }
          p._1(literal(0)) := count.read
        }
        scoped {
          val limit = local("limit", literal(1))
          val count = local("count", literal(0))
          gpuFor("i", literal(0), limit.read, 3) { _ =>
            count := count.read + literal(1)
            limit := literal(8)
          }
          p._1(literal(1)) := count.read
        }
        val sum = local("sum", literal(0))
        gpuFor("i", literal(0), literal(8), 3) { i =>
          gpuFor("j", literal(0), literal(4), 2) { j => accumulate(sum, i * literal(10) + j) }
        }
        p._1(literal(2)) := sum.read
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "changing.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      try
        val function = module.function(generated).toOption.get
        val out = context.allocate[Int](3).toOption.get
        try
          assertEquals(function.launch(definition.bind(Tuple1(out)), LaunchConfig(Grid.x(1), LaunchBlock.x(1))), Right(()))
          assertEquals(context.synchronize(), Right(()))
          assertEquals(out.copyToArray().toOption.get.toVector, Vector(1, 3, 186))
        finally out.close()
      finally module.close()
    }
