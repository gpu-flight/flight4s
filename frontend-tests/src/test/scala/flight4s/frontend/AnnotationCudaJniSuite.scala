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

  test("nested branches snapshot only active lanes and preserve both alternative paths"):
    val definition = PrototypeKernels.nestedBranches
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- Vector(0, 1, 63, 64, 65, 193) do
          val size = count + 32
          val initial = Array.tabulate(size)(i => i * 3 - 17)
          val data = context.allocate[Int](size).toOption.get
          val saved = context.allocate[Int](size).toOption.get
          val current = context.allocate[Int](size).toOption.get
          try
            for explicit <- Vector(false, true) do
              assertEquals(data.copyFrom(initial), Right(()))
              assertEquals(saved.copyFrom(Array.fill(size)(-999)), Right(()))
              assertEquals(current.copyFrom(Array.fill(size)(-999)), Right(()))
              launch(context, function, definition.bind((data, saved, current, count)),
                LaunchConfig(Grid.x(math.max(1, (count + 63) / 64)), LaunchBlock.x(64)), stream, explicit)
              assertEquals(data.copyToArray().toOption.get.toVector,
                Vector.tabulate(count)(900 + _) ++ initial.drop(count).toVector)
              assertEquals(saved.copyToArray().toOption.get.toVector,
                Vector.tabulate(count)(i => initial(i) + (if i % 2 == 0 then 1 else -1)) ++ Vector.fill(32)(-999))
              assertEquals(current.copyToArray().toOption.get.toVector,
                Vector.tabulate(count)(900 + _) ++ Vector.fill(32)(-999))
          finally
            current.close()
            saved.close()
            data.close()
      finally stream.close()
    }

  test("loop and composed foreach snapshots refresh on every device iteration"):
    for traversal <- Vector(false, true) do
      val definition = if traversal then PrototypeKernels.nestedTraversal else PrototypeKernels.nestedLoops
      withKernel(definition) { (context, function) =>
        val stream = context.createStream().toOption.get
        try
          for count <- Vector(0, 1, 65, 193); rounds <- Vector(0, 1, 2, 4) do
            val size = count + 32
            val initial = Array.tabulate(size)(i => i * 3 - 17)
            val increments = if traversal then
              (0 until rounds).map(_ + 1).filter(_ % 2 == 1).flatMap(value => Vector(value, value + 1)).toVector
            else (1 to rounds).toVector
            val data = context.allocate[Int](size).toOption.get
            val saved = context.allocate[Int](size).toOption.get
            try
              for explicit <- Vector(false, true) do
                assertEquals(data.copyFrom(initial), Right(()))
                assertEquals(saved.copyFrom(Array.fill(size)(-999)), Right(()))
                launch(context, function, definition.bind((data, saved, count, rounds)),
                  LaunchConfig(Grid.x(math.max(1, (count + 63) / 64)), LaunchBlock.x(64)), stream, explicit)
                assertEquals(data.copyToArray().toOption.get.toVector,
                  initial.take(count).map(_ + increments.sum).toVector ++ initial.drop(count).toVector,
                  s"traversal=$traversal count=$count rounds=$rounds explicit=$explicit")
                val expectedSaved = if increments.isEmpty then Vector.fill(count)(-999)
                  else initial.take(count).map(_ + increments.dropRight(1).sum).toVector
                assertEquals(saved.copyToArray().toOption.get.toVector, expectedSaved ++ Vector.fill(32)(-999))
            finally
              saved.close()
              data.close()
        finally stream.close()
      }

  test("mutable Int branches preserve snapshots and isolate same-name device locals"):
    val definition = PrototypeKernels.mutableBranches
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- Vector(0, 1, 63, 64, 65, 193) do
          val size = count + 32
          val initial = Array.tabulate(size)(i => i * 3 - 17)
          val data = context.allocate[Int](size).toOption.get
          val current = context.allocate[Int](size).toOption.get
          val saved = context.allocate[Int](size).toOption.get
          val inner = context.allocate[Int](size).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for explicit <- Vector(false, true) do
              assertEquals(current.copyFrom(Array.fill(size)(-999)), Right(()))
              assertEquals(saved.copyFrom(Array.fill(size)(-999)), Right(()))
              assertEquals(inner.copyFrom(Array.fill(size)(-999)), Right(()))
              launch(context, function, definition.bind((data, current, saved, inner, count)),
                LaunchConfig(Grid.x(math.max(1, (count + 63) / 64)), LaunchBlock.x(64)), stream, explicit)
              assertEquals(current.copyToArray().toOption.get.toVector,
                Vector.tabulate(count)(i => initial(i) + (if i % 2 == 0 then 3 else -2)) ++ Vector.fill(32)(-999))
              assertEquals(saved.copyToArray().toOption.get.toVector, initial.take(count).toVector ++ Vector.fill(32)(-999))
              assertEquals(inner.copyToArray().toOption.get.toVector,
                Vector.tabulate(count)(101 + _) ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            inner.close()
            saved.close()
            current.close()
            data.close()
      finally stream.close()
    }

  test("mutable loop-carried state and per-iteration locals execute zero one and multiple rounds"):
    val definition = PrototypeKernels.mutableLoops
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- Vector(0, 1, 65, 193); rounds <- Vector(0, 1, 2, 4) do
          val size = count + 32
          val initial = Array.tabulate(size)(i => i * 3 - 17)
          val increments = (0 until rounds).map(_ + 2)
          val data = context.allocate[Int](size).toOption.get
          val current = context.allocate[Int](size).toOption.get
          val previous = context.allocate[Int](size).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for explicit <- Vector(false, true) do
              assertEquals(current.copyFrom(Array.fill(size)(-999)), Right(()))
              assertEquals(previous.copyFrom(Array.fill(size)(-999)), Right(()))
              launch(context, function, definition.bind((data, current, previous, count, rounds)),
                LaunchConfig(Grid.x(math.max(1, (count + 63) / 64)), LaunchBlock.x(64)), stream, explicit)
              assertEquals(current.copyToArray().toOption.get.toVector,
                initial.take(count).map(_ + increments.sum).toVector ++ Vector.fill(32)(-999),
                s"count=$count rounds=$rounds explicit=$explicit")
              val expectedPrevious = if rounds == 0 then Vector.fill(count)(-999)
                else initial.take(count).map(_ + increments.dropRight(1).sum).toVector
              assertEquals(previous.copyToArray().toOption.get.toVector, expectedPrevious ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            previous.close()
            current.close()
            data.close()
      finally stream.close()
    }

  test("mutable Float row sums integrate with foreach and preserve the previous iteration snapshot"):
    val definition = PrototypeKernels.mutableRowSums
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for rows <- Vector(0, 1, 63, 64, 65, 193); columns <- Vector(0, 1, 3, 17) do
          val size = rows + 32
          val initial = Array.tabulate(rows * columns + 32)(i => (i % 13 - 6).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val sums = context.allocate[Float](size).toOption.get
          val previous = context.allocate[Float](size).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for explicit <- Vector(false, true) do
              assertEquals(sums.copyFrom(Array.fill(size)(-999f)), Right(()))
              assertEquals(previous.copyFrom(Array.fill(size)(-999f)), Right(()))
              launch(context, function, definition.bind((data, sums, previous, rows, columns)),
                LaunchConfig(Grid.x(math.max(1, (rows + 63) / 64)), LaunchBlock.x(64)), stream, explicit)
              val expected = Vector.tabulate(rows)(row =>
                (0 until columns).foldLeft(0.0f)((sum, column) => sum + initial(row * columns + column)))
              assertEquals(sums.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f),
                s"rows=$rows columns=$columns explicit=$explicit")
              val expectedPrevious = if columns == 0 then Vector.fill(rows)(-999f) else
                Vector.tabulate(rows)(row => (0 until columns - 1).foldLeft(0.0f)((sum, column) =>
                  sum + initial(row * columns + column)))
              assertEquals(previous.copyToArray().toOption.get.toVector, expectedPrevious ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            previous.close()
            sums.close()
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
        generated.compilerOptions, Vector(generated)), context.computeCapability, "annotation_prototype.cu")
        .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
      val module = context.load(artifact).toOption.get
      try body(context, module.function(generated).toOption.get)
      finally module.close()
    finally context.close()
