package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.dsl.ProductFoldState
import flight4s.core.ir.Expr
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaProductTraversalJniSuite extends FunSuite:
  private case class Entry(index: Expr[Int], weight: Expr[Float]) derives ProductFoldState
  private case class State(count: Expr[Int], total: Expr[Float], previous: Expr[Int]) derives ProductFoldState

  test("named products compose with tuples scalars nested generators and simultaneous state on the GPU"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      val definition = kernel("productTraversals", params(input[Int]("limits"), output[Int]("out"),
        output[Float]("totals"), value[Int]("rows"))) { p =>
        val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
        when(row < p._4) {
          val limit = p._1(row).read
          val state = gpuRange("i", literal(0), limit).by(2)
            .map(i => Entry(i, convert.i32ToF32(i)))
            .filter(e => (e.index > literal(0)) && ((e.index % literal(3)) === literal(0)))
            .map(e => Entry(literal(120) / e.index, e.weight + literal(0.5f)))
            .foldLeft("state", State(literal(0), literal(1.5f), literal(0))) { (s, e) =>
              State(s.count + literal(1), s.total + e.weight, s.previous + s.count + e.index)
            }
          val entries = for
            outer <- gpuRange("outer", literal(0), limit).by(3).map(i => Entry(i, literal(1.0f)))
            if outer.index > literal(0)
            inner <- gpuRange("inner", literal(0), outer.index % literal(4))
              .map(j => Entry(j, outer.weight))
          yield Entry(inner.index, convert.i32ToF32(outer.index + inner.index))
          val nested = entries.map(e => (e.index, e.weight))
            .flatMap { case (j, weight) =>
              gpuRange("repeat", literal(0), j).map(k => Entry(k, weight))
            }.flatMap(e => gpuRange("once", literal(0), literal(1)).map(_ => e.weight + convert.i32ToF32(e.index)))
            .foldLeft("nested", literal(0.0f))(_ + _)
          val current = local("current", literal(1))
          gpuRange("live", literal(0), literal(1)).map(i => Entry(i, convert.i32ToF32(current.read)))
            .foreach { e =>
              current := literal(7)
              p._3(row * literal(3) + literal(2)) := e.weight
            }
          p._2(row * literal(2)) := state.count
          p._2(row * literal(2) + literal(1)) := state.previous
          p._3(row * literal(3)) := state.total
          p._3(row * literal(3) + literal(1)) := nested
        }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
        generated.compilerOptions, Vector(generated)), context.computeCapability, "product_traversals.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val function = context.load(artifact).toOption.get.function(generated).toOption.get
      val counts = Array.tabulate(129)(i => Vector(-3, 0, 1, 2, 3, 4, 5, 31, 32, 33, 65, 127, 255)(i % 13))
      val limits = context.allocate[Int](counts.length).toOption.get
      val out = context.allocate[Int](counts.length * 2 + 32).toOption.get
      val totals = context.allocate[Float](counts.length * 3 + 32).toOption.get
      val stream = context.createStream().toOption.get
      assertEquals(limits.copyFrom(counts), Right(()))
      val states = counts.toVector.map { n =>
        (0 until n by 2).filter(i => i > 0 && i % 3 == 0)
          .map(i => (120 / i, i.toFloat + 0.5f))
          .foldLeft((0, 1.5f, 0)) { case (s, (quotient, weight)) =>
            (s._1 + 1, s._2 + weight, s._3 + s._1 + quotient)
          }
      }
      val expected = states.flatMap(s => Vector(s._1, s._3))
      val expectedTotals = counts.toVector.zip(states).flatMap { (n, s) =>
        val nested = (for
          i <- 0 until n by 3
          if i > 0
          j <- 0 until i % 4
          k <- 0 until j
        yield (i + j + k).toFloat).sum
        Vector(s._2, nested, 7.0f)
      }
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(counts.length * 2 + 32)(Int.MinValue)), Right(()))
        assertEquals(totals.copyFrom(Array.fill(counts.length * 3 + 32)(-123.0f)), Right(()))
        val arguments = definition.bind((limits, out, totals, counts.length))
        val config = LaunchConfig(Grid.x(5), LaunchBlock.x(32))
        assertEquals(if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        assertEquals(out.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(Int.MinValue))
        assertEquals(totals.copyToArray().toOption.get.toVector, expectedTotals ++ Vector.fill(32)(-123.0f))
      assertEquals(expected.size + expectedTotals.size, 645)
    finally context.close()
