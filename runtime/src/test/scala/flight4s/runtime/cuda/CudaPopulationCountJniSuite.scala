package flight4s.runtime.cuda

import munit.FunSuite
import scala.reflect.ClassTag
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaPopulationCountJniSuite extends FunSuite:
  test("population count matches host bits for signed unsigned and functional inputs"):
    withContext { context =>
      check[Int](context, identity)(using I32, summon[CudaHostCodec[Int]], summon[ClassTag[Int]])
      check[UInt](context, UInt.fromBits)(using U32, summon[CudaHostCodec[UInt]], summon[ClassTag[UInt]])
    }

  test("population count composes with ballot masks to compute selected counts and lane ranks"):
    withContext { context =>
      val count = 1031
      val random = new scala.util.Random(276)
      val values = Array.tabulate(count) { i =>
        if i >= 1024 then i & 1
        else (i / 32) % 4 match
          case 0 => 0
          case 1 => 1
          case 2 => i & 1
          case _ => random.nextInt()
      }
      val definition = kernel("ballot_counts", params(input[Int]("words"), output[Int]("out"), value[Int]("count"))) { p =>
        val (words, out, count) = p
        val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
        val lane = let("lane", threadIdx.x & literal(31))
        val full = literal(UInt.fromBits(-1))
        val selected = warp.ballot("selected", full, (index < count) && ((words(index).read & literal(1)) === literal(1)))
        when(index < count) {
          out(index) := bits.popCount(selected)
          out(count + index) := bits.popCount(selected & (~(full << lane)))
        }
      }
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "ballot_counts.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      try
        val function = module.function(generated).toOption.get
        val words = context.allocate[Int](count).toOption.get
        val out = context.allocate[Int](count * 2).toOption.get
        try
          assertEquals(words.copyFrom(values), Right(()))
          assertEquals(function.launch(definition.bind((words, out, count)),
            LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))), Right(()))
          assertEquals(context.synchronize(), Right(()))
          val actual = out.copyToArray().toOption.get
          for i <- 0 until count do
            val start = (i / 32) * 32
            val selected = (start until math.min(start + 32, count)).filter(j => (values(j) & 1) == 1)
            assertEquals(actual(i), selected.size, s"count at $i")
            assertEquals(actual(count + i), selected.count(_ < i), s"rank at $i")
        finally
          words.close()
          out.close()
      finally module.close()
    }

  private def check[T](context: CudaContext, fromBits: Int => T)(using
      bitwise: BitwiseType[T], codec: CudaHostCodec[T], tag: ClassTag[T]
  ): Unit =
    val count = 1031
    val random = new scala.util.Random(176)
    val edges = Array(0, 1, -1, Int.MinValue, Int.MaxValue, 0x55555555, 0xaaaaaaaa)
    val a = Array.tabulate(count)(i => if i < edges.length * edges.length then edges(i / edges.length) else random.nextInt())
    val b = Array.tabulate(count)(i => if i < edges.length * edges.length then edges(i % edges.length) else random.nextInt())
    val definition = kernel("population_count", params(input[T]("words"), input[T]("masks"), output[Int]("out"), value[Int]("count"))) { p =>
      val (words, masks, out, count) = p
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < count) {
        val word = words(index).read
        out(index) := bits.popCount(word)
        out(count + index) := bits.popCount(~word)
        out(count * literal(2) + index) := bits.popCount(word & masks(index).read)
        out(count * literal(3) + index) := bits.popCount(bits.popCount(word))
        val total = gpuRange("offset", literal(0), literal(4))
          .map(offset => bits.popCount(words((index + offset) % count).read))
          .filter(count => count > literal(0)).foldLeft("total", literal(0))(_ + _)
        out(count * literal(4) + index) := total
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated))
    val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "population_count.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).fold(failure => fail(failure.message), identity)
    try
      val function = module.function(generated).toOption.get
      val words = context.allocate[T](count).toOption.get
      val masks = context.allocate[T](count).toOption.get
      val out = context.allocate[Int](count * 5).toOption.get
      try
        assertEquals(words.copyFrom(a.map(fromBits)), Right(()))
        assertEquals(masks.copyFrom(b.map(fromBits)), Right(()))
        assertEquals(function.launch(definition.bind((words, masks, out, count)),
          LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))), Right(()))
        assertEquals(context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        for i <- 0 until count do
          val total = (0 until 4).map(offset => Integer.bitCount(a((i + offset) % count))).filter(_ > 0).sum
          val expected = Vector(Integer.bitCount(a(i)), Integer.bitCount(~a(i)), Integer.bitCount(a(i) & b(i)),
            Integer.bitCount(Integer.bitCount(a(i))), total)
          expected.zipWithIndex.foreach { (value, operation) =>
            assertEquals(actual(operation * count + i), value, s"${bitwise.cudaName} i=$i operation=$operation")
          }
      finally
        words.close()
        masks.close()
        out.close()
    finally module.close()

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
