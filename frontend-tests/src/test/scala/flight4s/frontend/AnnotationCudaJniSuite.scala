package flight4s.frontend

import scala.annotation.experimental
import munit.FunSuite
import flight4s.frontend.examples.PrototypeKernels
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.ir.{Kernel, KernelInvocation}
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.*

@experimental
class AnnotationCudaJniSuite extends FunSuite:
  test("annotated vector addition preserves bounds tails inputs and both stream paths"):
    val definition = PrototypeKernels.vectorAdd
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- Vector(0, 1, 63, 64, 65, 193, 257) do
          val size = count + 64
          val leftValues = Array.tabulate(size)(i => i.toFloat * 0.5f)
          val rightValues = Array.tabulate(size)(i => 10f - i.toFloat * 0.25f)
          val left = context.allocate[Float](size).toOption.get
          val right = context.allocate[Float](size).toOption.get
          val target = context.allocate[Float](size).toOption.get
          try
            assertEquals(left.copyFrom(leftValues), Right(()))
            assertEquals(right.copyFrom(rightValues), Right(()))
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(size)(-999f)), Right(()))
              val config = LaunchConfig(Grid.x(math.max(1, (count + 63) / 64)), LaunchBlock.x(64))
              launch(context, function, definition.bind((left, right, target, count)), config, stream, explicit)
              val expected = Vector.tabulate(count)(i => leftValues(i) + rightValues(i)) ++ Vector.fill(64)(-999f)
              assertEquals(target.copyToArray().toOption.get.toVector, expected, s"count=$count explicit=$explicit")
              assertEquals(left.copyToArray().toOption.get.toVector, leftValues.toVector)
              assertEquals(right.copyToArray().toOption.get.toVector, rightValues.toVector)
          finally
            target.close()
            right.close()
            left.close()
      finally stream.close()
    }

  test("ordinary Int vals preserve earlier reads and aliases after writes on the GPU"):
    val definition = PrototypeKernels.intSnapshots
    withKernel(definition) { (context, function) =>
      val size = 224
      val initial = Array.tabulate(size)(i => i * 3 - 17)
      val data = context.allocate[Int](size).toOption.get
      val saved = context.allocate[Int](size).toOption.get
      val current = context.allocate[Int](size).toOption.get
      val stream = context.createStream().toOption.get
      try
        for explicit <- Vector(false, true) do
          assertEquals(data.copyFrom(initial), Right(()))
          assertEquals(saved.copyFrom(Array.fill(size)(-999)), Right(()))
          assertEquals(current.copyFrom(Array.fill(size)(-999)), Right(()))
          launch(context, function, definition.bind((data, saved, current)),
            LaunchConfig(Grid.x(3), LaunchBlock.x(64)), stream, explicit)
          assertEquals(data.copyToArray().toOption.get.toVector,
            Vector.tabulate(192)(900 + _) ++ initial.drop(192).toVector)
          assertEquals(saved.copyToArray().toOption.get.toVector,
            initial.take(192).map(_ * 2).toVector ++ Vector.fill(32)(-999))
          assertEquals(current.copyToArray().toOption.get.toVector,
            Vector.tabulate(192)(900 + _) ++ Vector.fill(32)(-999))
      finally
        stream.close()
        current.close()
        saved.close()
        data.close()
    }

  test("ordinary Float vals preserve original values across writes on the GPU"):
    val definition = PrototypeKernels.floatSnapshots
    withKernel(definition) { (context, function) =>
      val size = 224
      val initial = Array.tabulate(size)(i => i.toFloat * 0.5f - 17f)
      val data = context.allocate[Float](size).toOption.get
      val saved = context.allocate[Float](size).toOption.get
      val stream = context.createStream().toOption.get
      try
        for explicit <- Vector(false, true) do
          assertEquals(data.copyFrom(initial), Right(()))
          assertEquals(saved.copyFrom(Array.fill(size)(-999f)), Right(()))
          launch(context, function, definition.bind((data, saved)),
            LaunchConfig(Grid.x(3), LaunchBlock.x(64)), stream, explicit)
          assertEquals(data.copyToArray().toOption.get.toVector,
            Vector.fill(192)(900.25f) ++ initial.drop(192).toVector)
          assertEquals(saved.copyToArray().toOption.get.toVector,
            initial.take(192).map(_ * 2f).toVector ++ Vector.fill(32)(-999f))
      finally
        stream.close()
        saved.close()
        data.close()
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
        generated.compilerOptions, Vector(generated)), context.computeCapability, "annotation_prototype.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      try body(context, module.function(generated).toOption.get)
      finally module.close()
    finally context.close()
