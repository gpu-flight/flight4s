package flight4s.runtime.cuda

import munit.FunSuite
import scala.reflect.ClassTag
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaBitwiseExpressionJniSuite extends FunSuite:
  test("CUDA bitwise operators and functional mask folds match signed and unsigned host bit patterns"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0) match
      case Right(context) => context
      case Left(failure) if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) => fail(failure.message)
    try
      check[Int](context, identity, identity)(using I32, summon[CudaHostCodec[Int]], summon[ClassTag[Int]])
      check[UInt](context, UInt.fromBits, _.toIntBits)(using U32, summon[CudaHostCodec[UInt]], summon[ClassTag[UInt]])
    finally context.close()

  private def check[T](context: CudaContext, fromBits: Int => T, bits: T => Int)(using
      bitwise: BitwiseType[T] & EqualityComparableType[T], codec: CudaHostCodec[T], tag: ClassTag[T]
  ): Unit =
    val count = 1031
    val random = new scala.util.Random(107)
    val edges = Array(0, 1, -1, Int.MinValue, Int.MaxValue, 0x55555555, 0xaaaaaaaa)
    val a = Array.tabulate(count)(i => if i < edges.length * edges.length then edges(i / edges.length) else random.nextInt())
    val b = Array.tabulate(count)(i => if i < edges.length * edges.length then edges(i % edges.length) else random.nextInt())
    val definition = kernel("bitwise", params(input[T]("a"), input[T]("b"), output[T]("out"), value[Int]("count"))) { p =>
      val (a, b, out, count) = p
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < count) {
        val left = a(index).read
        val right = b(index).read
        out(index) := left & right
        out(count + index) := left | right
        out(count * literal(2) + index) := left ^ right
        out(count * literal(3) + index) := (left & right) | (left ^ right)
        val combined = gpuRange("offset", literal(0), literal(4))
          .map(offset => a((index + offset) % count).read ^ b((index + offset) % count).read)
          .filter(value => (value & literal(fromBits(1))) === literal(fromBits(1)))
          .foldLeft("combined", literal(fromBits(0)))((acc, value) => acc | value)
        out(count * literal(4) + index) := combined
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(
      generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
    )
    val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "bitwise.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).fold(failure => fail(failure.message), identity)
    try
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val left = context.allocate[T](count).toOption.get
      val right = context.allocate[T](count).toOption.get
      val out = context.allocate[T](count * 5).toOption.get
      try
        assertEquals(left.copyFrom(a.map(fromBits)), Right(()))
        assertEquals(right.copyFrom(b.map(fromBits)), Right(()))
        assertEquals(function.launch(definition.bind((left, right, out, count)),
          LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))), Right(()))
        assertEquals(context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        for i <- 0 until count do
          val folded = (0 until 4).map(offset => a((i + offset) % count) ^ b((i + offset) % count))
            .filter(value => (value & 1) == 1).foldLeft(0)(_ | _)
          val expected = Vector(a(i) & b(i), a(i) | b(i), a(i) ^ b(i), (a(i) & b(i)) | (a(i) ^ b(i)), folded)
          expected.zipWithIndex.foreach { (value, operation) =>
            assertEquals(bits(actual(operation * count + i)), value, s"${bitwise.cudaName} i=$i operation=$operation")
          }
      finally
        left.close()
        right.close()
        out.close()
    finally module.close()
