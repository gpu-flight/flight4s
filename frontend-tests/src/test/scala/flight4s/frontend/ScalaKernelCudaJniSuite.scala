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

  test("quoted row sums execute serial column loops with empty work tails and unchanged inputs"):
    val definition = ScalaKernels.rowSum
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for rows <- counts; columns <- Vector(0, 1, 3, 17, 65) do
          val initial = Array.tabulate(math.max(1, rows * columns))(i => (i % 17 - 8).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](rows + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(rows)(row =>
              (0 until columns).foldLeft(0.0f)((sum, column) => sum + initial(row * columns + column)))
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(rows + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, rows, columns)), config(rows), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted ranges snapshot mutable bounds and preserve negative empty reversed and Int-edge ranges"):
    val definition = ScalaKernels.rangeBounds
    val bounds = Vector((0, 0), (2, 2), (4, 1), (-3, 2), (2, 7),
      (Int.MinValue, Int.MinValue + 3), (Int.MaxValue - 3, Int.MaxValue), (Int.MaxValue, Int.MaxValue))
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val visits = context.allocate[Int](count + 32).toOption.get
          val last = context.allocate[Int](count + 32).toOption.get
          try
            for (from, until) <- bounds; explicit <- Vector(false, true) do
              assertEquals(visits.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              assertEquals(last.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((visits, last, count, from, until)), config(count), stream, explicit)
              val iterations = math.max(0L, until.toLong - from.toLong).toInt
              assertEquals(visits.copyToArray().toOption.get.toVector,
                Vector.fill(count)(iterations) ++ Vector.fill(32)(-999))
              assertEquals(last.copyToArray().toOption.get.toVector,
                Vector.fill(count)(if iterations == 0 then 123 else until - 1) ++ Vector.fill(32)(-999))
          finally
            last.close()
            visits.close()
      finally stream.close()
    }

  test("quoted nested loops resolve shadowed indices and refresh snapshots each outer iteration"):
    val definition = ScalaKernels.nestedRanges
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 7 - 3)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rounds <- Vector(0, 1, 2, 4); explicit <- Vector(false, true) do
              val expected = initial.take(count).map { item =>
                var total = item
                for outer <- 0 until rounds do
                  val before = total
                  for inner <- 0 until outer do total = total + before + inner
                total
              }.toVector
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, rounds)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted chained guards protect tight row inputs and retain ordered Float sums"):
    val definition = ScalaKernels.guardedRows
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for rows <- counts; columns <- Vector(0, 1, 3, 17, 65) do
          val initial = Array.tabulate(math.max(1, rows * columns))(i => (i % 17 - 8).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](rows + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for threshold <- Vector(-4.0f, 0.0f, 4.0f); explicit <- Vector(false, true) do
              val expected = Vector.tabulate(rows)(row =>
                (0 until columns).foldLeft(0.0f) { (sum, column) =>
                  val item = initial(row * columns + column)
                  if item > threshold then sum + item else sum
                })
              assertEquals(target.copyFrom(Array.fill(rows + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, rows, columns, threshold)), config(rows), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted withFilter guards refresh mutable state without changing captured bounds"):
    val definition = ScalaKernels.guardedState
    val bounds = Vector((0, 0), (2, 2), (4, 1), (-3, 2), (-3, 7), (0, 8), (2, 7))
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 7 - 3)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for (from, until) <- bounds; explicit <- Vector(false, true) do
              val expected = initial.take(count).map { item =>
                var total = item
                for index <- from until until do
                  if index >= total && index % 2 == 0 then total = total + index + 1
                total
              }.toVector
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, from, until)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted guarded nested generators preserve dependent bounds snapshots and shadowed indices"):
    val definition = ScalaKernels.nestedGuards
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 7 - 3)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rounds <- Vector(0, 1, 2, 4, 7); explicit <- Vector(false, true) do
              val expected = initial.take(count).map { item =>
                var total = item
                for outer <- 0 until rounds do
                  if outer % 2 == 0 then
                    for inner <- 0 until outer do total = total + outer + inner
                for outer <- 1 until rounds do
                  for inner <- 0 until outer do
                    if inner % 2 == 0 then total += inner
                total
              }.toVector
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, rounds)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("staged Float yield plans preserve lazy guarded reads map order and both stream paths"):
    val definition = ScalaKernels.yieldRows
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for rows <- counts; columns <- Vector(0, 1, 3, 17, 65) do
          val initial = Array.tabulate(math.max(1, rows * columns))(i => (i % 17 - 8).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](rows + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for threshold <- Vector(-4.0f, 0.0f, 4.0f); explicit <- Vector(false, true) do
              val expected = Vector.tabulate(rows)(row =>
                (0 until columns).foldLeft(0.0f) { (sum, column) =>
                  val item = initial(row * columns + column)
                  if item > threshold then sum + item * 2.0f else sum
                })
              assertEquals(target.copyFrom(Array.fill(rows + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, rows, columns, threshold)), config(rows), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("reused yield plans refresh live captures but retain bounds and mapped values after buffer writes"):
    val definition = ScalaKernels.yieldReuse
    val bounds = Vector((0, 0), (2, 2), (4, 1), (-3, 2), (-3, 7), (0, 8), (2, 7))
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(count + 32)(i => i % 7 - 3)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](initial.length).toOption.get
          try
            for (from, until) <- bounds; explicit <- Vector(false, true) do
              val expected = initial.take(count).map { initial =>
                var data = initial
                var total = 0
                for bias <- Vector(1, 3); index <- from until until if index % 2 == 0 do
                  val item = data + index + bias
                  data = item + 1
                  total += item + item
                (data, total)
              }.toVector
              assertEquals(data.copyFrom(initial), Right(()))
              assertEquals(target.copyFrom(Array.fill(initial.length)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, from, until)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected.map(_._2) ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, expected.map(_._1) ++ initial.drop(count).toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("nested staged maps retain lexical captures live predicates and loop-local refresh"):
    val definition = ScalaKernels.yieldNested
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 7 - 3)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rounds <- Vector(0, 1, 2, 4, 7); explicit <- Vector(false, true) do
              val expected = initial.take(count).map { item =>
                var total = item
                for outer <- 0 until rounds do
                  val before = total
                  for index <- 0 until outer do
                    val value = index + before
                    if value >= total then total += value + outer
                total
              }.toVector
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, rounds)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("ordered Float folds preserve cancellation order seeds and lazy guarded tight reads"):
    val definition = ScalaKernels.foldRows
    val pattern = Vector(1.0e20f, 1.0f, -1.0e20f, 3.0f, -2.0f, 0.25f, -0.5f)
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for rows <- counts; columns <- Vector(0, 1, 3, 17, 65) do
          val initial = Array.tabulate(math.max(1, rows * columns))(i => pattern(i % pattern.size))
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](rows + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for threshold <- Vector(-2.0e20f, 0.0f, 2.0e20f); seed <- Vector(0.0f, 7.0f); explicit <- Vector(false, true) do
              val expected = Vector.tabulate(rows)(row =>
                (0 until columns).foldLeft(seed) { (sum, column) =>
                  val item = initial(row * columns + column)
                  if item > threshold then sum - item * 2.0f else sum
                })
              assertEquals(target.copyFrom(Array.fill(rows + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, rows, columns, threshold, seed)), config(rows), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("ordered reused Int folds retain fixed bounds fresh seeds live data and independent results"):
    val definition = ScalaKernels.foldReuse
    val bounds = Vector((0, 0), (2, 2), (4, 1), (-3, 2), (-3, 7), (0, 8), (2, 7))
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(count + 32)(i => i % 7 - 3)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](initial.length).toOption.get
          try
            for (from, until) <- bounds; explicit <- Vector(false, true) do
              val expected = initial.take(count).map { original =>
                val first = (from until until).filter(_ % 2 == 0)
                  .foldLeft(original)((sum, index) => sum - (original + index + 1) - (original + index + 1))
                val second = (from until until).filter(_ % 2 == 0).foldLeft(first + 1) { (sum, index) =>
                  val item = first + index + 3
                  if item > sum then item else sum - item
                }
                (first, second + first)
              }.toVector
              assertEquals(data.copyFrom(initial), Right(()))
              assertEquals(target.copyFrom(Array.fill(initial.length)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, from, until)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected.map(_._2) ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, expected.map(_._1) ++ initial.drop(count).toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("ordered nested Boolean and Double folds refresh each iteration and preserve shadows"):
    val definition = ScalaKernels.foldNested
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 17 - 8).toDouble * 0.25)
          val data = context.allocate[Double](initial.length).toOption.get
          val target = context.allocate[Double](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rounds <- Vector(0, 1, 2, 4, 7); enabled <- Vector(false, true); explicit <- Vector(false, true) do
              val expected = initial.take(count).map { item =>
                var total = item
                for outer <- 0 until rounds do
                  val values = (0 until outer).map(_ + 1)
                  val accepted = values.foldLeft(enabled)((found, item) => found || item % 2 == 0)
                  total = values.foldLeft(total)((state, item) => if accepted then state / 2.0 else state - 1.0) + 0.25
                total
              }.toVector
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              launch(context, function, definition.bind((data, target, count, rounds, enabled)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999.0))
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
