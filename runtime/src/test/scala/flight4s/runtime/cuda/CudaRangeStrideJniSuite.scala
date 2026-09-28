package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CompilerOptions, CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Kernel
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.Float16

class CudaRangeStrideJniSuite extends FunSuite:
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

  private def withFunction[Args <: Tuple](
      context: CudaContext,
      definition: Kernel[Args],
      options: CompilerOptions = CompilerOptions()
  )(
      body: CudaFunction[Args] => Unit
  ): Unit =
    val generated = CudaCodegen.generate(definition, options).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
    val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "range_stride.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).fold(failure => fail(failure.message), identity)
    try body(module.function(generated).toOption.get)
    finally module.close()

  test("strided range composition executes sums folds guards and dependent nested generators"):
    withContext { context =>
      val rows = 259
      val columns = 17
      val bounds = Vector.tabulate(rows)(row => Vector((0, 17), (2, 2), (16, 0), (1, 16), (2, 17))(row % 5))
      val data = Array.tabulate(rows * columns)(index => Vector(1e20f, 1f, -1e20f, 3f, -2f, 0.5f)(index % 6))
      val indices = bounds.map { (from, until) => (from until until by 3).toVector }
      val definition = kernel("composition", params(input[Float]("source"), input[Int]("starts"), input[Int]("ends"),
        output[Float]("sums"), output[Int]("stats"), value[Int]("rows"))) { p =>
        val (source, starts, ends, sums, stats, rowCount) = p
        val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
        when(row < rowCount) {
          val range = gpuRange("i", starts(row).read, ends(row).read).by(3)
          sums(row) := range.map(i => source(row * literal(columns) + i).read).sum(literal(2.0f))
          val filtered = range.map(_ * literal(2)).filter(_ < literal(20))
            .foldLeft("ordered", literal(1))((acc, i) => acc * literal(3) + i)
          stats(row * literal(6)) := filtered
          val pair = range.foldLeft("pair", (literal(0), literal(0))) { (state, i) =>
            (state._1 + i, state._2 + literal(1))
          }
          stats(row * literal(6) + literal(1)) := pair._1
          stats(row * literal(6) + literal(2)) := pair._2
          val nested = for
            i <- range
            j <- gpuRange("j", literal(0), i).by(2)
            if j < literal(5)
          yield i + j
          stats(row * literal(6) + literal(3)) := nested.foldLeft("nested", literal(7))(_ + _)
          val visits = local("visits", literal(0))
          range.foreach(_ => accumulate(visits, literal(1)))
          stats(row * literal(6) + literal(4)) := visits.read
          stats(row * literal(6) + literal(5)) := range.map { i =>
            gpuRange("inner", literal(0), i).by(2).map(j => i + j).sum(literal(0))
          }.sum(literal(0))
        }
      }
      withFunction(context, definition) { function =>
        val source = context.allocate[Float](data.length).toOption.get
        val starts = context.allocate[Int](rows).toOption.get
        val ends = context.allocate[Int](rows).toOption.get
        val sums = context.allocate[Float](rows).toOption.get
        val stats = context.allocate[Int](rows * 6).toOption.get
        try
          assertEquals(source.copyFrom(data), Right(()))
          assertEquals(starts.copyFrom(bounds.map(_._1).toArray), Right(()))
          assertEquals(ends.copyFrom(bounds.map(_._2).toArray), Right(()))
          assertEquals(function.launch(definition.bind((source, starts, ends, sums, stats, rows)),
            LaunchConfig(Grid.x(3), LaunchBlock.x(128))), Right(()))
          assertEquals(context.synchronize(), Right(()))
          val expectedSums = indices.zipWithIndex.map((items, row) => items.foldLeft(2.0f)((acc, i) => acc + data(row * columns + i)))
          assertEquals(sums.copyToArray().toOption.get.toVector, expectedSums)
          val expectedStats = indices.flatMap { items =>
            val nested = for i <- items; j <- 0 until i by 2 if j < 5 yield i + j
            Vector(items.map(_ * 2).filter(_ < 20).foldLeft(1)((acc, i) => acc * 3 + i),
              items.sum, items.size, nested.sum + 7, items.size,
              items.map(i => (0 until i by 2).map(j => i + j).sum).sum)
          }
          assertEquals(stats.copyToArray().toOption.get.toVector, expectedStats)
        finally
          stats.close()
          sums.close()
          ends.close()
          starts.close()
          source.close()
      }
    }

  test("strided reductions handle Int endpoints without counter wraparound"):
    withContext { context =>
      Vector(1, 2, 3, 16, 31, 32, Int.MaxValue).foreach { step =>
        val random = new scala.util.Random(step.toLong)
        val bounds = Vector((0, 0), (8, 0), (-8, 8), (0, 9),
          (Int.MaxValue - 2, Int.MaxValue), (Int.MaxValue - 1, Int.MaxValue),
          (Int.MinValue, Int.MinValue + 3), (Int.MaxValue, Int.MaxValue)) ++
          (if step == Int.MaxValue then Vector((Int.MinValue, Int.MaxValue)) else Vector.empty) ++
          Vector.fill(257) {
            val from = random.nextInt()
            (from, math.min(Int.MaxValue.toLong, from.toLong + step.toLong * random.nextInt(7) + random.nextInt(2)).toInt)
          }
        val expected = bounds.map { (from, until) =>
          Iterator.iterate(from.toLong)(_ + step.toLong).takeWhile(_ < until.toLong).map(_.toInt).toVector
        }
        val definition = kernel("endpoints", params(input[Int]("starts"), input[Int]("ends"),
          output[Int]("out"), value[Int]("rows"))) { p =>
          val (starts, ends, out, rows) = p
          val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
          when(row < rows) {
            val range = gpuRange("i", starts(row).read, ends(row).read).by(step)
            out(row * literal(2)) := range.map(bits.popCount(_)).sum(literal(7))
            out(row * literal(2) + literal(1)) := range.map(_ => literal(1)).sum(literal(0))
          }
        }
        withFunction(context, definition) { function =>
          val starts = context.allocate[Int](bounds.size).toOption.get
          val ends = context.allocate[Int](bounds.size).toOption.get
          val out = context.allocate[Int](bounds.size * 2).toOption.get
          try
            assertEquals(starts.copyFrom(bounds.map(_._1).toArray), Right(()))
            assertEquals(ends.copyFrom(bounds.map(_._2).toArray), Right(()))
            assertEquals(function.launch(definition.bind((starts, ends, out, bounds.size)),
              LaunchConfig(Grid.x((bounds.size + 127) / 128), LaunchBlock.x(128))), Right(()))
            assertEquals(context.synchronize(), Right(()))
            assertEquals(out.copyToArray().toOption.get.toVector,
              expected.flatMap(items => Vector(7 + items.map(java.lang.Integer.bitCount).sum, items.size)), s"step=$step")
          finally
            out.close()
            ends.close()
            starts.close()
        }
      }
    }

  test("strided half input sums accumulate in Float"):
    withContext { context =>
      val cudaRoot = sys.env.getOrElse("CUDA_PATH", fail("set CUDA_PATH to the toolkit root for the half-header GPU test"))
      val include = java.nio.file.Path.of(cudaRoot, "include")
      assert(java.nio.file.Files.isRegularFile(include.resolve("cuda_fp16.h")))
      val options = CompilerOptions(additionalNvrtcOptions = Vector(s"--include-path=$include"))
      val definition = kernel("half_sum", params(input[Float16]("source"), output[Float]("out"))) { p =>
        p._2(literal(0)) := gpuRange("i", literal(0), literal(17)).by(3)
          .map(i => p._1(i).read).sum(literal(0.5f))
      }
      withFunction(context, definition, options) { function =>
        val source = context.allocate[Float16](17).toOption.get
        val out = context.allocate[Float](1).toOption.get
        try
          assertEquals(source.copyFrom(Array.fill(17)(Float16.fromBits(0x3c00.toShort))), Right(()))
          assertEquals(function.launch(definition.bind((source, out)), LaunchConfig(Grid.x(1), LaunchBlock.x(1))), Right(()))
          assertEquals(context.synchronize(), Right(()))
          assertEquals(out.copyToArray().toOption.get.toVector, Vector(6.5f))
        finally
          out.close()
          source.close()
      }
    }
