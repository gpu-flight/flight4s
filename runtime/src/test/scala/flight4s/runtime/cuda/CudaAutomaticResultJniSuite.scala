package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.ProductFoldState
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, Kernel, MemoryOrder}
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.UInt

private case class GpuResultState(a: Expr[Int], b: Expr[Int]) derives ProductFoldState

class CudaAutomaticResultJniSuite extends FunSuite:
  test("unnamed folds preserve ordered scalar and simultaneous structured state updates"):
    withContext { context =>
      val definition = kernel("automaticFolds", params(output[Int]("out"))) { p =>
        val i = let(blockIdx.x * blockDim.x + threadIdx.x)
        val count = i % literal(7)
        val range = gpuRange(literal(0), count)
        val scalar = range.foldLeft(literal(100))(_ - _)
        val pair = range.foldLeft((literal(1), literal(2)))((s, _) => (s._2, s._1))
        val tuple = range.foldLeft((literal(1), literal(2), literal(3)))((s, _) => (s._2, s._3, s._1))
        val product = range.foldLeft(GpuResultState(literal(0), literal(1)))((s, _) => GpuResultState(s.b, s.a + s.b))
        val tupleTotal = range.map(x => (x, x + literal(1))).foldLeft(literal(0))((s, x) => s + x._1 + x._2)
        val productTotal = range.map(x => GpuResultState(x, x)).foldLeft(literal(0))((s, x) => s - x.a)
        val ordered = range.filter(x => (x % literal(2)) === literal(0))
          .flatMap(x => gpuRange(literal(0), x + literal(1))).orderedSum(literal(10))
        val expression = range.map(identity).sum(literal(0))
        val values = Vector(scalar, pair._1, pair._2, tuple._1, tuple._2, tuple._3,
          product.a, product.b, tupleTotal, productTotal, ordered, expression)
        values.zipWithIndex.foreach { (value, column) => p._1(i * literal(12) + literal(column)) := value }
      }
      val expected = (0 until 192).toVector.flatMap { i =>
        val n = i % 7
        val sum = (0 until n).sum
        val pair = if n % 2 == 0 then Vector(1, 2) else Vector(2, 1)
        val tuple = Vector(1, 2, 3).drop(n % 3) ++ Vector(1, 2, 3).take(n % 3)
        val product = (0 until n).foldLeft((0, 1))((s, _) => (s._2, s._1 + s._2))
        Vector(100 - sum) ++ pair ++ tuple ++ Vector(product._1, product._2, n * n, -sum,
          10 + (0 until n).filter(_ % 2 == 0).flatMap(x => 0 to x).sum, sum)
      }
      verify(context, definition, expected)
    }

  test("unnamed warp operations execute for every supported group width"):
    withContext { context =>
      for width <- Vector(1, 2, 4, 8, 16, 32) do
        val definition = kernel("automaticWarp", params(output[Int]("out"))) { p =>
          val i = let(blockIdx.x * blockDim.x + threadIdx.x)
          val mask = literal(UInt.fromBits(-1))
          val delta = literal(UInt.fromBits(1))
          val direct = if width == 32 then warp.shuffle(mask, i, literal(0)) else warp.shuffle(mask, i, literal(0), width)
          val up = if width == 32 then warp.shuffleUp(mask, i, delta) else warp.shuffleUp(mask, i, delta, width)
          val down = if width == 32 then warp.shuffleDown(mask, i, delta) else warp.shuffleDown(mask, i, delta, width)
          val selector = literal(if width == 1 then 0 else 1)
          val xor = if width == 32 then warp.shuffleXor(mask, i, selector) else warp.shuffleXor(mask, i, selector, width)
          val ballot = bits.popCount(warp.ballot(mask, (i % literal(32)) < literal(16)))
          val all = choose(warp.all(mask, threadIdx.x < literal(64)))(literal(1))(literal(0))
          val any = choose(warp.any(mask, (i % literal(32)) === literal(0)))(literal(1))(literal(0))
          val sum = if width == 32 then warp.reduceSum(UInt.fromBits(-1), i) else warp.reduceSum(UInt.fromBits(-1), i, width)
          val difference = if width == 32 then warp.reduceTree(UInt.fromBits(-1), i)(_ - _)
            else warp.reduceTree(UInt.fromBits(-1), i, width)(_ - _)
          Vector(direct, up, down, xor, ballot, all, any, sum, difference).zipWithIndex.foreach {
            (value, column) => p._1(i * literal(9) + literal(column)) := value
          }
        }
        val expected = (0 until 192).toVector.flatMap { i =>
          val base = i / width * width
          Vector(base, if i % width == 0 then i else i - 1,
            if i % width == width - 1 then i else i + 1,
            if width == 1 then i else i ^ 1, 16, 1, 1,
            (base until base + width).sum, differenceTree((base until base + width).toVector))
        }
        verify(context, definition, expected)
    }

  test("unnamed atomics return stable old values across operations orders and storage scopes"):
    import MemoryOrder.*
    withContext { context =>
      for storage <- Vector("device", "blockGlobal", "shared") do
        val definition = kernel("automaticAtomics", params(output[Int]("out"))) { p =>
          val i = let(blockIdx.x * blockDim.x + threadIdx.x)
          val base = i * literal(15)
          val scratch = sharedArray[Int](64)
          val ref = storage match
            case "device" => atomic.device(p._1(base))
            case "blockGlobal" => atomic.block(p._1(base))
            case _ => atomic.block(scratch(threadIdx.x))
          ref.store(literal(12), Release)
          val loaded = ref.load(Acquire)
          val exchanged = ref.exchange(literal(20), Relaxed)
          val added = ref.fetchAdd(literal(3), AcquireRelease)
          val subtracted = ref.fetchSub(literal(4), Release)
          val min = ref.fetchMin(literal(5), Relaxed)
          val max = ref.fetchMax(literal(9), Relaxed)
          val and = ref.fetchAnd(literal(6), Relaxed)
          val or = ref.fetchOr(literal(10), Relaxed)
          val xor = ref.fetchXor(literal(3), Relaxed)
          val success = ref.compareExchange(literal(9), literal(4), SequentiallyConsistent, Acquire)
          val failure = ref.compareExchange(literal(9), literal(7), AcquireRelease, Acquire)
          val legacy = if storage == "shared" then atomicFetchAdd(scratch(threadIdx.x), literal(2))
            else atomicFetchAdd(p._1(base), literal(2))
          val current = ref.load(Acquire)
          p._1(base) := current
          Vector(loaded, exchanged, added, subtracted, min, max, and, or, xor, success, failure, legacy, current, loaded)
            .zipWithIndex.foreach { (value, column) => p._1(base + literal(column + 1)) := value }
        }
        verify(context, definition, Vector.fill(192)(Vector(6, 12, 12, 20, 23, 19, 5, 9, 0, 10, 9, 4, 4, 6, 12)).flatten)
    }

  private def differenceTree(values: Vector[Int]): Int =
    if values.size == 1 then values.head
    else differenceTree(values.grouped(2).map(pair => pair(0) - pair(1)).toVector)

  private def verify(context: CudaContext,
      definition: Kernel[Tuple1[flight4s.core.ir.DeviceBuffer[Int]]], expected: Vector[Int]): Unit =
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated)), context.computeCapability, "automatic_results.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).toOption.get
    val out = context.allocate[Int](expected.size + 32).toOption.get
    val stream = context.createStream().toOption.get
    try
      val function = module.function(generated).toOption.get
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(expected.size + 32)(-999)), Right(()))
        val args = definition.bind(Tuple1(out))
        val config = LaunchConfig(Grid.x(3), LaunchBlock.x(64))
        assertEquals(if explicit then function.launch(args, config, stream) else function.launch(args, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        assertEquals(out.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999), s"stream=$explicit")
    finally
      stream.close()
      out.close()
      module.close()

  private def withContext(body: CudaContext => Unit): Unit =
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try body(context)
    finally context.close()
