package flight4s.runtime.cuda

import java.nio.charset.StandardCharsets

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.compiler.NvrtcArtifact
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{Expr, Kernel}
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.unsafe.raw.RawCuda

class CudaResourcesJniSuite extends FunSuite:
  private val nativeLibraryConfigured =
    sys.props.contains("flight4s.cuda.native.path")

  test("repeated scoped helpers preserve ordered GPU stores and outer-local updates"):
    assume(nativeLibraryConfigured, "set flight4s.cuda.native.path to run JNI tests")
    val count = 257
    val values = Array.tabulate(count)(_.toFloat)
    val definition = kernel("scopedTransforms",
      params(inOut[Float]("data"), output[Int]("counts"), value[Int]("n"))) { bindings =>
      val (data, counts, n) = bindings
      val index = local("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index.read < n) {
        val updates = local("updates", literal(1))
        def transform(f: Expr[Float] => Expr[Float])(using BlockBuilder): Unit = scoped {
          val temporary = local("temporary", data(index.read).read)
          data(index.read) := f(temporary.read)
          updates := updates.read + literal(1)
        }
        transform(_ * literal(2.0f))
        transform(_ + literal(3.0f))
        counts(index.read) := updates.read
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(
      generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
    )
    val context = openContext()
    try
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "scoped_helpers.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val data = context.allocate[Float](count).toOption.get
      val counts = context.allocate[Int](count).toOption.get
      assertEquals(data.copyFrom(values), Right(()))
      assertEquals(counts.copyFrom(Array.fill(count)(-1)), Right(()))
      assertEquals(function.launch(
        definition.bind((data, counts, count)),
        LaunchConfig(Grid.x((count + 127) / 128), LaunchBlock.x(128))
      ), Right(()))
      assertEquals(context.synchronize(), Right(()))
      assertEquals(data.copyToArray().toOption.get.toVector, values.map(_ * 2.0f + 3.0f).toVector)
      assertEquals(counts.copyToArray().toOption.get.toVector, Vector.fill(count)(3))
    finally context.close()

  test("staged range composition executes row sums and foreach with empty-range semantics"):
    assume(nativeLibraryConfigured, "set flight4s.cuda.native.path to run JNI tests")
    val rows = 259
    val columns = 7
    val values = Array.tabulate(rows * columns)(i => (i % 11).toFloat)
    val signature = params(
      input[Float]("source"), output[Float]("sums"), output[Float]("squares"),
      value[Int]("rowCount"), value[Int]("from"), value[Int]("until")
    )
    val definition = kernel("rowSquares", signature) { bindings =>
      val (source, sums, squares, rowCount, from, until) = bindings
      val row = local("row", blockIdx.x * blockDim.x + threadIdx.x)
      when(row.read < rowCount) {
        sums(row.read) := gpuRange("column", from, until)
          .map(column => source(row.read * literal(columns) + column).read)
          .map(value => value * value)
          .sum(literal(3.0f))
        gpuRange("copyColumn", from, until).foreach { column =>
          val index = row.read * literal(columns) + column
          val value = source(index).read
          squares(index) := value * value
        }
      }
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val generatedModule = GeneratedCudaModule(
      generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)
    )
    val context = openContext()
    try
      val artifact = NvrtcCompiler.compile(generatedModule, context.computeCapability, "row_squares.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(generated).fold(failure => fail(failure.message), identity)
      val source = context.allocate[Float](values.length).toOption.get
      val sums = context.allocate[Float](rows).toOption.get
      val squares = context.allocate[Float](values.length).toOption.get
      assertEquals(source.copyFrom(values), Right(()))
      for (from, until) <- Vector((0, columns), (2, 6), (4, 4), (5, 2)) do
        assertEquals(sums.copyFrom(Array.fill(rows)(-1.0f)), Right(()))
        assertEquals(squares.copyFrom(Array.fill(values.length)(-1.0f)), Right(()))
        assertEquals(
          function.launch(
            definition.bind((source, sums, squares, rows, from, until)),
            LaunchConfig(Grid.x((rows + 127) / 128), LaunchBlock.x(128))
          ),
          Right(())
        )
        assertEquals(context.synchronize(), Right(()))
        val expectedSums = Vector.tabulate(rows) { row =>
          (from until until).foldLeft(3.0f) { (sum, column) =>
            val value = values(row * columns + column)
            sum + value * value
          }
        }
        val expectedSquares = values.indices.map { i =>
          if i % columns >= from && i % columns < until then values(i) * values(i)
          else -1.0f
        }.toVector
        assertEquals(sums.copyToArray().toOption.get.toVector, expectedSums)
        assertEquals(squares.copyToArray().toOption.get.toVector, expectedSquares)
    finally context.close()

  test("raw vectorAdd compiles loads launches and matches the CPU result"):
    assume(nativeLibraryConfigured, "set flight4s.cuda.native.path to run JNI tests")
    val raw = RawCuda.kernel(
      "rawVectorAdd",
      params(input[Float]("a"), input[Float]("b"), output[Float]("out"), value[Int]("n")),
      """extern "C" __global__ void rawVectorAdd(
        |    const float* a, const float* b, float* out, int n) {
        |  int i = blockIdx.x * blockDim.x + threadIdx.x;
        |  if (i < n) out[i] = a[i] + b[i];
        |}
        |""".stripMargin,
      CompilerOptions(), KernelLaunchRequirements()
    )
    val context = openContext()
    try
      val artifact = NvrtcCompiler.compile(raw, context.computeCapability, "raw_vector_add.cu") match
        case Right(value) => value
        case Left(failure) => fail(failure.message + "\n" + failure.compileLog)
      val module = context.load(artifact).fold(failure => fail(failure.message), identity)
      val function = module.function(raw).fold(failure => fail(failure.message), identity)
      val stream = context.createStream().toOption.get
      val count = 513
      val a = Array.tabulate(count)(_.toFloat)
      val b = Array.tabulate(count)(i => (i * 2).toFloat)
      val left = context.allocate[Float](count).toOption.get
      val right = context.allocate[Float](count).toOption.get
      val out = context.allocate[Float](count).toOption.get
      assertEquals(left.copyFrom(a), Right(()))
      assertEquals(right.copyFrom(b), Right(()))
      assertEquals(
        function.launch(
          raw.bind((left, right, out, count)),
          LaunchConfig(Grid.x((count + 255) / 256), LaunchBlock.x(256)), stream
        ),
        Right(())
      )
      module.close()
      assert(!function.isValid)
      assertEquals(stream.synchronize(), Right(()))
      val actual = out.copyToArray().toOption.get
      assertEquals(actual.toVector, a.zip(b).map(_ + _).toVector)
    finally context.close()

  test("NVRTC artifact loads and resolves a typed CUDA function"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val context = openContext()
    var loadedModule = Option.empty[CudaModule]
    try
      val generated = generatedFixture("loadedKernel")
      val artifact = NvrtcCompiler.compile(
        generated.module,
        context.computeCapability,
        "loaded_kernel.cu"
      ) match
        case Right(value) => value
        case Left(failure) =>
          fail(failure.message + "\n" + failure.compileLog)

      val module = context.load(artifact) match
        case Right(value) => value
        case Left(failure) =>
          fail(failure.message + "\n" + failure.errorLog)
      loadedModule = Some(module)

      val function = module.function(generated.kernel) match
        case Right(value) => value
        case Left(failure) => fail(failure.message)

      assert(context.isOpen)
      assert(module.isOpen)
      assert(function.isValid)
      assertEquals(function.name, "loadedKernel")
      assert(function.signature eq generated.kernel.signature)
      assert(function.nativeHandle != 0L)
      assert(function.attributes.maxThreadsPerBlock > 0)
      assert(function.attributes.staticSharedMemoryBytes >= 0)
      assert(function.attributes.constantMemoryBytes >= 0)
      assert(function.attributes.localMemoryBytes >= 0)
      assert(function.attributes.registersPerThread >= 0)

      function.launch(
        generated.definition.bind(EmptyTuple),
        LaunchConfig(Grid.x(1), LaunchBlock.x(1))
      ) match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)

      module.close()
      assert(!module.isOpen)
      assert(!function.isValid)
      context.synchronize() match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)
      loadedModule = None
    finally
      loadedModule.foreach { module =>
        if module.isOpen then module.close()
      }
      if context.isOpen then context.close()

  test("Driver module failures retain JIT diagnostics"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val context = openContext()
    try
      val generated = generatedFixture("invalidPtxKernel")
      val validArtifact = NvrtcCompiler.compile(
        generated.module,
        context.computeCapability,
        "invalid_ptx_kernel.cu"
      ).toOption.get
      val invalidArtifact = validArtifact.copy(
        ptx = IArray.unsafeFromArray(
          "not valid PTX".getBytes(StandardCharsets.UTF_8)
        )
      )

      val failure =
        context.load(invalidArtifact).swap.toOption.get

      assertEquals(failure.operation, "CUDA PTX module load")
      assert(failure.resultCode != 0)
      assert(failure.resultName.startsWith("CUDA_ERROR_"))
      assert(failure.errorLog.nonEmpty)
    finally
      if context.isOpen then context.close()

  test("typed vectorAdd executes through an owned explicit stream"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val elementCount = 1024
    val leftValues =
      Array.tabulate(elementCount)(index => index.toFloat)
    val rightValues =
      Array.tabulate(elementCount)(index => (index * 2).toFloat)
    val expected =
      Array.tabulate(elementCount) { index =>
        leftValues(index) + rightValues(index)
      }

    val leftParam = input[Float]("left")
    val rightParam = input[Float]("right")
    val outputParam = output[Float]("output")
    val countParam = value[Int]("elementCount")
    val definition = kernel(
      "vectorAdd",
      params(leftParam, rightParam, outputParam, countParam)
    ) { bindings =>
      val left = bindings.head
      val right = bindings.tail.head
      val output = bindings.tail.tail.head
      val count = bindings.tail.tail.tail.head
      val index = local(
        "index",
        blockIdx.x * blockDim.x + threadIdx.x
      )
      when(index.read < count) {
        output(index.read) :=
          left(index.read).read + right(index.read).read
      }
    }
    val generatedKernel = CudaCodegen.generate(definition) match
      case Right(value) => value
      case Left(error) => fail(error.message)
    val generatedModule = GeneratedCudaModule(
      cudaSource = generatedKernel.cudaSource,
      sourceMap = generatedKernel.sourceMap,
      compilerOptions = generatedKernel.compilerOptions,
      kernels = Vector(generatedKernel)
    )

    val context = openContext()
    try
      val artifact = NvrtcCompiler.compile(
        generatedModule,
        context.computeCapability,
        "vector_add.cu"
      ) match
        case Right(value) => value
        case Left(failure) =>
          fail(failure.message + "\n" + failure.compileLog)
      val module = context.load(artifact) match
        case Right(value) => value
        case Left(failure) => fail(failure.message)
      val function = module.function(generatedKernel) match
        case Right(value) => value
        case Left(failure) => fail(failure.message)
      val stream = context.createStream().toOption.get
      val completed = context.createEvent().toOption.get
      val leftHost = context.allocatePinned[Float](elementCount).toOption.get
      val rightHost = context.allocatePinned[Float](elementCount).toOption.get
      val outputHost = context.allocatePinned[Float](elementCount).toOption.get
      val leftBuffer = context.allocate[Float](elementCount).toOption.get
      val rightBuffer = context.allocate[Float](elementCount).toOption.get
      val outputBuffer = context.allocate[Float](elementCount).toOption.get

      leftHost.copyFrom(leftValues)
      rightHost.copyFrom(rightValues)
      assertEquals(leftBuffer.copyFromAsync(leftHost, stream), Right(()))
      assertEquals(rightBuffer.copyFromAsync(rightHost, stream), Right(()))

      val invocation = definition.bind(
        (leftBuffer, rightBuffer, outputBuffer, elementCount)
      )
      function.launch(
        invocation,
        LaunchConfig(
          grid = Grid.x(elementCount / 256),
          block = LaunchBlock.x(256)
        ),
        stream
      ) match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)

      outputBuffer.copyToAsync(outputHost, stream) match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)
      completed.record(stream) match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)
      leftHost.close()
      rightHost.close()
      leftBuffer.close()
      rightBuffer.close()
      outputBuffer.close()
      module.close()
      assert(!leftHost.isOpen)
      assert(!rightHost.isOpen)
      assert(!leftBuffer.isOpen)
      assert(!rightBuffer.isOpen)
      assert(!outputBuffer.isOpen)
      assert(!module.isOpen)
      completed.query() match
        case Right(_) => ()
        case Left(failure) => fail(failure.message)
      completed.synchronize() match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)
      assertEquals(completed.query(), Right(true))
      val actual = outputHost.toArray
      assertEquals(actual.toSeq, expected.toSeq)
    finally
      if context.isOpen then context.close()

  test("dynamic shared-memory block reduction matches the CPU sum"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val elementCount = 256
    val inputValues =
      Array.tabulate(elementCount)(index => (index % 8).toFloat)
    val expected = inputValues.sum
    val inputParam = input[Float]("input")
    val outputParam = output[Float]("output")
    val definition = kernel(
      "blockReduceSum",
      params(inputParam, outputParam)
    ) { bindings =>
      val input = bindings.head
      val output = bindings.tail.head
      val scratch = dynamicSharedArray[Float]("scratch")

      scratch(threadIdx.x) := input(threadIdx.x).read
      barrier()
      Vector(128, 64, 32, 16, 8, 4, 2, 1).foreach { stride =>
        when(threadIdx.x < literal(stride)) {
          scratch(threadIdx.x) :=
            scratch(threadIdx.x).read +
              scratch(threadIdx.x + literal(stride)).read
        }
        barrier()
      }
      when(threadIdx.x === literal(0)) {
        output(literal(0)) := scratch(literal(0)).read
      }
    }
    val generatedKernel = CudaCodegen.generate(definition) match
      case Right(value) => value
      case Left(error) => fail(error.message)
    val generatedModule = GeneratedCudaModule(
      cudaSource = generatedKernel.cudaSource,
      sourceMap = generatedKernel.sourceMap,
      compilerOptions = generatedKernel.compilerOptions,
      kernels = Vector(generatedKernel)
    )

    val context = openContext()
    try
      val artifact = NvrtcCompiler.compile(
        generatedModule,
        context.computeCapability,
        "block_reduce_sum.cu"
      ) match
        case Right(value) => value
        case Left(failure) =>
          fail(failure.message + "\n" + failure.compileLog)
      val module = context.load(artifact) match
        case Right(value) => value
        case Left(failure) => fail(failure.message)
      val function = module.function(generatedKernel) match
        case Right(value) => value
        case Left(failure) => fail(failure.message)
      val inputBuffer = context.allocate[Float](elementCount).toOption.get
      val outputBuffer = context.allocate[Float](1).toOption.get

      assertEquals(inputBuffer.copyFrom(inputValues), Right(()))
      function.launch(
        definition.bind((inputBuffer, outputBuffer)),
        LaunchConfig(
          grid = Grid.x(1),
          block = LaunchBlock.x(elementCount),
          dynamicSharedMemoryBytes = elementCount * java.lang.Float.BYTES
        )
      ) match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)

      module.close()
      inputBuffer.close()
      assertEquals(context.synchronize(), Right(()))
      val actual = outputBuffer.copyToArray() match
        case Right(value) => value
        case Left(failure) => fail(failure.message)
      assertEquals(actual.toSeq, Seq(expected))
    finally
      if context.isOpen then context.close()

  test("partial pinned transfers preserve untouched elements on the GPU"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val context = openContext()
    try
      val initial = context.allocatePinned[Int](8).toOption.get
      val source = context.allocatePinned[Int](6).toOption.get
      val destination = context.allocatePinned[Int](7).toOption.get
      val device = context.allocate[Int](8).toOption.get
      val stream = context.createStream().toOption.get

      initial.copyFrom(Array.fill(8)(0))
      source.copyFrom(Array(10, 20, 30, 40, 50, 60))
      destination.copyFrom(Array.fill(7)(-1))
      assertEquals(device.copyFromAsync(initial, stream), Right(()))
      assertEquals(
        device.copyFromAsync(source, 1, 3, 3, stream),
        Right(())
      )
      assertEquals(
        device.copyToAsync(destination, 2, 1, 4, stream),
        Right(())
      )
      initial.close()
      source.close()
      device.close()
      assert(!initial.isOpen)
      assert(!source.isOpen)
      assert(!device.isOpen)
      assertEquals(stream.synchronize(), Right(()))
      assertEquals(
        destination.toArray.toSeq,
        Seq(-1, 0, 20, 30, 40, -1, -1)
      )
    finally
      if context.isOpen then context.close()

  private def openContext(): CudaContext =
    CudaContext.open(0) match
      case Right(context) => context
      case Left(failure)
          if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) =>
        fail(failure.message)

  private def generatedFixture(
      name: String
  ): GeneratedFixture =
    val definition = kernel(name, params()) { _ => () }
    val generatedKernel = CudaCodegen.generate(definition) match
      case Right(value) => value
      case Left(error) => fail(error.message)
    GeneratedFixture(
      definition,
      generatedKernel,
      GeneratedCudaModule(
        cudaSource = generatedKernel.cudaSource,
        sourceMap = generatedKernel.sourceMap,
        compilerOptions = generatedKernel.compilerOptions,
        kernels = Vector(generatedKernel)
      )
    )

  private final case class GeneratedFixture(
      definition: Kernel[EmptyTuple],
      kernel: GeneratedKernel[EmptyTuple],
      module: GeneratedCudaModule
  )
