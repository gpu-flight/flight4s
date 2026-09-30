package flight4s.frontend

import scala.annotation.experimental
import munit.FunSuite
import flight4s.frontend.examples.ScalaKernels
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.ir.{Kernel, KernelInvocation}
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.*

@experimental
class ScalaKernelCudaJniSuite extends FunSuite:
  private val counts = Vector(0, 1, 63, 64, 65, 193, 257)
  private def config(count: Int): LaunchConfig =
    LaunchConfig(Grid.x(math.max(1, (count + 63) / 64)), LaunchBlock.x(64))

  test("named Scala Float parameters scale guarded lanes and preserve inputs tails and both stream paths"):
    val definition = ScalaKernels.scale
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val size = count + 32
          val initial = Array.tabulate(size)(i => (i - 90).toFloat * 0.25f)
          val data = context.allocate[Float](size).toOption.get
          val target = context.allocate[Float](size).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for explicit <- Vector(false, true); factor <- Vector(0.0f, 2.0f, -0.5f) do
              assertEquals(target.copyFrom(Array.fill(size)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count, factor)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector,
                initial.take(count).map(_ * factor).toVector ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("Scala Int vars snapshot old data and isolate shadowed branch locals on the GPU"):
    val definition = ScalaKernels.branches
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val size = count + 32
          val initial = Array.tabulate(size)(i => i % 17 - 8)
          val data = context.allocate[Int](size).toOption.get
          val saved = context.allocate[Int](size).toOption.get
          val target = context.allocate[Int](size).toOption.get
          try
            for explicit <- Vector(false, true) do
              assertEquals(data.copyFrom(initial), Right(()))
              assertEquals(saved.copyFrom(Array.fill(size)(-999)), Right(()))
              assertEquals(target.copyFrom(Array.fill(size)(-999)), Right(()))
              launch(context, function, definition.bind((data, saved, target, count)), config(count), stream, explicit)
              val current = Vector.tabulate(count)(i => initial(i) + (if i % 2 == 0 then 3 else -2))
              val expected = Vector.tabulate(count)(i =>
                if initial(i) < 0 then initial(i) + 7 else if current(i) > 0 then current(i) else initial(i))
              assertEquals(data.copyToArray().toOption.get.toVector, current ++ initial.drop(count).toVector)
              assertEquals(saved.copyToArray().toOption.get.toVector, initial.take(count).toVector ++ Vector.fill(32)(-999))
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
          finally
            target.close()
            saved.close()
            data.close()
      finally stream.close()
    }

  test("Scala short-circuit conditions do not read out of range on inactive GPU lanes"):
    val definition = ScalaKernels.shortCircuit
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val size = math.max(1, count)
          val initial = Array.tabulate(size)(i => i % 7 - 3)
          val data = context.allocate[Int](size).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector,
                initial.take(count).map(value => if value == 0 then 11 else value + 2).toVector ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("Scala Double values and Boolean kernel arguments select branches with zero work and disabled execution"):
    val definition = ScalaKernels.doubles
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val size = count + 32
          val initial = Array.tabulate(size)(i => (i % 17 - 8).toDouble * 0.25)
          val data = context.allocate[Double](size).toOption.get
          val target = context.allocate[Double](size).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for explicit <- Vector(false, true); enabled <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(size)(-999.0)), Right(()))
              launch(context, function, definition.bind((data, target, count, enabled, 0.5)), config(count), stream, explicit)
              val expected = if enabled then initial.take(count).map(value =>
                if value < 0.0 then (value + 0.5) * 2.0 else (value + 0.5) / 2.0).toVector ++ Vector.fill(32)(-999.0)
              else Vector.fill(size)(-999.0)
              assertEquals(target.copyToArray().toOption.get.toVector, expected)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  private def launch[Args <: Tuple](context: CudaContext, function: CudaFunction[Args],
      invocation: KernelInvocation[Args], config: LaunchConfig, stream: CudaStream, explicit: Boolean): Unit =
    assertEquals(if explicit then function.launch(invocation, config, stream) else function.launch(invocation, config), Right(()))
    assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))

  private def withKernel[Args <: Tuple](definition: Kernel[Args])(
      body: (CudaContext, CudaFunction[Args]) => Unit): Unit =
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
      val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
        generated.compilerOptions, Vector(generated)), context.computeCapability, "scala_kernel_prototype.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      try body(context, module.function(generated).toOption.get)
      finally module.close()
    finally context.close()
