package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CompilerOptions, CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaFloatVectorJniSuite extends FunSuite:
  test("native float2 and float4 buffers survive shared local dynamic and pinned storage"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(f => fail(f.message), identity)
    try
      val bits = Vector(0x80000000, 0, 0x7fc01234, 0x7f800000, 0xff800000, 1, 0x3f800000, 0xbf800000)
      def component(i: Int): Float = java.lang.Float.intBitsToFloat(bits(i % bits.size))
      checkStorage(context, Array.tabulate(197)(i => Float2(component(i), component(i + 1))),
        Float2(9f, 10f), (v: Float2) => Vector(v.x, v.y))
      checkStorage(context, Array.tabulate(197)(i => Float4(component(i), component(i + 1), component(i + 2), component(i + 3))),
        Float4(9f, 10f, 11f, 12f), (v: Float4) => Vector(v.x, v.y, v.z, v.w))
    finally context.close()

  test("vector aggregates component maps zips conditionals and snapshots execute on GPU"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(f => fail(f.message), identity)
    try
      val definition = kernel("vectorArithmetic", params(input[Float4]("source"), output[Float4]("out"), value[Int]("count"))) { p =>
        val i = let("i", blockIdx.x * blockDim.x + threadIdx.x)
        when(i < p._3) {
          val v = local("v", p._1(i).read)
          val snapshot = let("snapshot", v.read)
          v := literal(Float4(100f, 101f, 102f, 103f))
          val pair = float2(snapshot.w, snapshot.z)
          val swizzle = float4(pair.x, pair.y, snapshot.y, snapshot.x)
          val result = swizzle.map(_ * literal(2f)).zipWith(snapshot)(_ + _)
          p._2(i) := choose(i < literal(100))(result)(literal(Float4(-0.0f, Float.PositiveInfinity, Float.NegativeInfinity, Float.NaN)))
        }
      }
      val generated = CudaCodegen.generate(definition, options).fold(f => fail(f.message), identity)
      val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
        generated.compilerOptions, Vector(generated)), context.computeCapability, "vector_arithmetic.cu")
        .fold(f => fail(f.message + "\n" + f.compileLog), identity)
      val module = context.load(artifact).toOption.get
      val stream = context.createStream().toOption.get
      val data = Array.tabulate(197)(i => Float4(i.toFloat, i + 0.25f, i - 0.5f, -i.toFloat))
      val source = context.allocate[Float4](data.length).toOption.get
      val out = context.allocate[Float4](data.length + 8).toOption.get
      val sentinel = Float4(-9f, -8f, -7f, -6f)
      try
        assertEquals(source.copyFrom(data), Right(()))
        val function = module.function(generated).toOption.get
        for explicit <- Vector(false, true) do
          assertEquals(out.copyFrom(Array.fill(data.length + 8)(sentinel)), Right(()))
          val config = LaunchConfig(Grid.x(4), LaunchBlock.x(64))
          val arguments = definition.bind((source, out, data.length))
          assertEquals(if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config), Right(()))
          assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
          val actual = out.copyToArray().toOption.get
          for i <- actual.indices do
            if i >= data.length then assertEquals(actual(i), sentinel)
            else if i >= 100 then
              assertEquals(java.lang.Float.floatToRawIntBits(actual(i).x), 0x80000000)
              assertEquals(actual(i).y, Float.PositiveInfinity)
              assertEquals(actual(i).z, Float.NegativeInfinity)
              assert(actual(i).w.isNaN)
            else
              val v = data(i)
              assertEquals(actual(i), Float4(2f * v.w + v.x, 2f * v.z + v.y, 2f * v.y + v.z, 2f * v.x + v.w))
      finally
        source.close()
        out.close()
        stream.close()
        module.close()
    finally context.close()

  private def options: CompilerOptions =
    val root = sys.env.getOrElse("CUDA_PATH", fail("CUDA_PATH is required for vector headers"))
    CompilerOptions(additionalNvrtcOptions = Vector(s"--include-path=${java.nio.file.Path.of(root, "include")}"))

  private def checkStorage[T](context: CudaContext, data: Array[T], sentinel: T, components: T => Vector[Float])(using
      vectorType: FloatVectorType[T], codec: CudaHostCodec[T], classTag: scala.reflect.ClassTag[T]
  ): Unit =
    val definition = kernel("vectorStorage", params(input[T]("source"), output[T]("out"), value[Int]("count"))) { p =>
      val tile = sharedArray[T]("tile", 64)
      val dynamic = dynamicSharedArray[T]("dynamic")
      val scratch = localArray[T]("scratch", 1)
      val i = let("i", blockIdx.x * blockDim.x + threadIdx.x)
      tile(threadIdx.x) := choose(i < p._3)(p._1(i).read)(literal(sentinel))
      barrier()
      dynamic(threadIdx.x) := tile(threadIdx.x).read
      barrier()
      scratch(literal(0)) := dynamic(threadIdx.x).read
      when(i < p._3) { p._2(i) := scratch(literal(0)).read }
    }
    val generated = CudaCodegen.generate(definition, options).fold(f => fail(f.message), identity)
    assertEquals(generated.launchRequirements.dynamicSharedMemory.get.elementSizeBytes, vectorType.sizeBytes)
    val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated)), context.computeCapability, "vector_storage.cu")
      .fold(f => fail(f.message + "\n" + f.compileLog), identity)
    val module = context.load(artifact).toOption.get
    val stream = context.createStream().toOption.get
    val source = context.allocate[T](data.length).toOption.get
    val out = context.allocate[T](data.length + 8).toOption.get
    val pinned = context.allocatePinned[T](data.length).toOption.get
    try
      assertEquals(source.sizeBytes, data.length.toLong * vectorType.sizeBytes)
      pinned.copyFrom(data)
      val function = module.function(generated).toOption.get
      for explicit <- Vector(false, true) do
        assertEquals(if explicit then source.copyFromAsync(pinned, stream) else source.copyFrom(pinned), Right(()))
        if explicit then assertEquals(stream.synchronize(), Right(()))
        assertEquals(out.copyFrom(Array.fill(data.length + 8)(sentinel)), Right(()))
        val config = LaunchConfig(Grid.x(4), LaunchBlock.x(64), dynamicSharedMemoryBytes = 64 * vectorType.sizeBytes)
        val arguments = definition.bind((source, out, data.length))
        assertEquals(if explicit then function.launch(arguments, config, stream) else function.launch(arguments, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        val actual = out.copyToArray().toOption.get
        for i <- actual.indices do
          val expected = if i < data.length then data(i) else sentinel
          assertEquals(components(actual(i)).map(java.lang.Float.floatToRawIntBits),
            components(expected).map(java.lang.Float.floatToRawIntBits), s"${vectorType.cudaName} stream=$explicit index=$i")
    finally
      pinned.close()
      source.close()
      out.close()
      stream.close()
      module.close()
