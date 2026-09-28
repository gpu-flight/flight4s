package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaTupleTraversalJniSuite extends FunSuite:
  test("tuple elements compose across maps guards nested generators and mixed-state folds"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      val definition = kernel("tupleTraversals", params(input[Int]("limits"), output[Int]("out"),
        output[Float]("totals"), value[Int]("rows"))) { p =>
        val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
        when(row < p._4) {
          val limit = p._1(row).read
          val state = gpuRange("i", literal(0), limit).by(2)
            .map(i => (i, convert.i32ToF32(i), (i % literal(3)) === literal(0)))
            .filter { case (i, _, selected) => selected && (i > literal(0)) }
            .map { case (i, x, _) => (literal(120) / i, x + literal(0.5f)) }
            .foldLeft("state", (literal(0), literal(1.5f), literal(0))) { case (s, (quotient, x)) =>
              (s._1 + literal(1), s._2 + x, s._3 + s._1 + quotient)
            }
          val tuples = for
            (i, next) <- gpuRange("outer", literal(0), limit).by(3).map(i => (i, i + literal(1)))
            if i > literal(0)
            (j, sum) <- gpuRange("inner", literal(0), i % literal(4)).map(j => (j, i + j))
          yield (i, next, j, sum)
          val nested = tuples.flatMap { case (_, next, j, sum) =>
            gpuRange("repeat", literal(0), j).map(k => next + sum + k)
          }.foldLeft("nested", literal(0))(_ + _)
          p._2(row * literal(3)) := state._1
          p._2(row * literal(3) + literal(1)) := state._3
          p._2(row * literal(3) + literal(2)) := nested
          p._3(row) := state._2
        }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
        generated.compilerOptions, Vector(generated)), context.computeCapability, "tuple_traversals.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val function = context.load(artifact).toOption.get.function(generated).toOption.get
      val counts = Array.tabulate(129)(i => Vector(-3, 0, 1, 2, 3, 4, 5, 31, 32, 33, 65, 127, 255)(i % 13))
      val limits = context.allocate[Int](counts.length).toOption.get
      val out = context.allocate[Int](counts.length * 3 + 32).toOption.get
      val totals = context.allocate[Float](counts.length + 32).toOption.get
      val stream = context.createStream().toOption.get
      assertEquals(limits.copyFrom(counts), Right(()))
      val expectedStates = counts.toVector.map { n =>
        (0 until n by 2).filter(i => i % 3 == 0 && i > 0)
          .map(i => (120 / i, i.toFloat + 0.5f))
          .foldLeft((0, 1.5f, 0)) { case (s, (quotient, x)) => (s._1 + 1, s._2 + x, s._3 + s._1 + quotient) }
      }
      val expected = counts.toVector.zip(expectedStates).flatMap { (n, s) =>
        val nested = (for
          i <- 0 until n by 3
          if i > 0
          j <- 0 until i % 4
          k <- 0 until j
        yield i + 1 + (i + j) + k).sum
        Vector(s._1, s._3, nested)
      }
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(counts.length * 3 + 32)(Int.MinValue)), Right(()))
        assertEquals(totals.copyFrom(Array.fill(counts.length + 32)(-123.0f)), Right(()))
        val arguments = definition.bind((limits, out, totals, counts.length))
        val config = LaunchConfig(Grid.x(5), LaunchBlock.x(32))
        assertEquals(if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        assertEquals(out.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(Int.MinValue))
        assertEquals(totals.copyToArray().toOption.get.toVector, expectedStates.map(_._2) ++ Vector.fill(32)(-123.0f))
      assertEquals(expected.size + expectedStates.size, 516)
    finally context.close()
