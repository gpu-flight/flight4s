package flight4s.frontend

import scala.annotation.experimental
import munit.FunSuite
import flight4s.frontend.examples.ScalaKernels
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.ir.{DeviceBuffer, Kernel, KernelInvocation}
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

  test("flatMap guards protect inner bound loads and preserve ordered Float cancellation on both streams"):
    val definition = ScalaKernels.flatMapRows
    val pattern = Vector(1.0e20f, 1.0f, -1.0e20f, 3.0f, -2.0f, 0.25f, -0.5f)
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for rows <- counts; columns <- Vector(0, 1, 3, 17); empty <- Vector(false, true) do
          val initial = Array.tabulate(math.max(1, rows * columns))(i => pattern(i % pattern.size))
          val lengths = Array.tabulate(math.max(1, columns))(i => if empty then 0 else Vector(-1, 0, 1, 3, 5)(i % 5))
          val data = context.allocate[Float](initial.length).toOption.get
          val sizes = context.allocate[Int](lengths.length).toOption.get
          val target = context.allocate[Float](rows + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            assertEquals(sizes.copyFrom(lengths), Right(()))
            for threshold <- Vector(-2.0e20f, 0.0f, 2.0e20f); explicit <- Vector(false, true) do
              val expected = Vector.tabulate(rows) { row =>
                var total = 7.0f
                for column <- 0 until columns; inner <- 0 until lengths(column) if inner % 2 == 0 do
                  val item = initial(row * columns + column) + (if inner == 0 then 0.0f else 0.5f)
                  if item > threshold then total = total - item * 2.0f
                total
              }
              assertEquals(target.copyFrom(Array.fill(rows + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, sizes, target, rows, columns, threshold)), config(rows), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
              assertEquals(sizes.copyToArray().toOption.get.toVector, lengths.toVector)
          finally
            target.close()
            sizes.close()
            data.close()
      finally stream.close()
    }

  test("reused flatMap refreshes inner bounds per outer item without rereading saved outer values"):
    val definition = ScalaKernels.flatMapReuse
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
                var data = original
                var limit = 3
                var total = 0
                for outer <- from until until if outer % 2 == 0 do
                  val mapped = data + outer
                  val end = limit
                  for inner <- 0 until end do
                    val item = mapped + inner
                    if item >= 0 then
                      data = item + 1
                      total += item + item
                      limit = 1
                for outer <- from until until if outer % 2 == 0 do
                  val mapped = data + outer
                  for inner <- 0 until 2 do
                    val item = mapped + inner
                    if item >= 0 then total -= item
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

  test("chained flatMap executes aliased inner plans with Boolean elements and global Double fold state"):
    val definition = ScalaKernels.flatMapNested
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
              val expected = initial.take(count).map { original =>
                var found = false
                var total = original
                for outer <- 0 until rounds; inner <- 0 until 2; index <- 0 until outer + inner do
                  val item = index + 1
                  val flag = if item % 2 == 0 then enabled else !enabled
                  found = found || flag
                  total = total / 2.0 - (if flag then 2.0 else 1.0)
                if found then total else original
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

  test("seven parameter tuples launch mixed scalar types with tails and both stream paths"):
    val definition = ScalaKernels.tupleScale
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
            for enabled <- Vector(false, true); cutoff <- Vector(-1.0, 0.0, 0.125)
                factor <- Vector(0.0f, 2.0f, -0.5f); bias <- Vector(0.0f, 7.0f); explicit <- Vector(false, true) do
              val expected = initial.take(count).map(x => if enabled && cutoff > 0.0 then x * factor + bias else -999f).toVector
              assertEquals(target.copyFrom(Array.fill(size)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count, factor, bias, enabled, cutoff)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("tuple row guards preserve lazy bounds mixed fields ordered Float state and input tails"):
    val definition = ScalaKernels.tupleRows
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for rows <- counts; columns <- Vector(0, 1, 3, 7) do
          val initial = Array.tabulate(math.max(1, rows * columns))(i => (i % 17 - 8).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](rows + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for threshold <- Vector(-2.0f, 0.0f, 2.0f); seed <- Vector(7.0f, -3.0f); explicit <- Vector(false, true) do
              val expected = Vector.tabulate(rows) { row =>
                var total = seed
                for column <- 0 until columns if column % 2 == 0 do
                  val item = initial(row * columns + column)
                  if item > threshold then total = total - item * 2.0f - (if column == 0 then 0.5f else 0.0f)
                total
              }
              assertEquals(target.copyFrom(Array.fill(rows + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, rows, columns, threshold, seed)), config(rows), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("tuple fields retain pre-store values while reused plans observe new data on the GPU"):
    checkTupleReuse(ScalaKernels.tupleReuse)

  test("named tuple snapshots retain pre-store values on both GPU streams"):
    checkTupleReuse(ScalaKernels.namedReuse)

  private def checkTupleReuse(definition: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int, Int, Int)]): Unit =
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(count + 32)(i => i % 7 - 3)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](initial.length).toOption.get
          try
            for (from, until) <- Vector((0, 0), (3, 1), (-3, 2), (0, 7), (2, 7)); explicit <- Vector(false, true) do
              val expected = initial.take(count).map { original =>
                var data = original
                var total = 0
                for i <- from until until do
                  val first = data + i
                  val saved = data
                  if first >= 0 then
                    data = first + 1
                    total += saved + saved + i
                for i <- from until until do
                  val first = data + i
                  val saved = data
                  if first >= 0 then total = total - first - saved
                (data, total)
              }.toVector
              assertEquals(data.copyFrom(initial), Right(()))
              assertEquals(target.copyFrom(Array.fill(initial.length)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, from, until)), config(count), stream, explicit)
              assertEquals(data.copyToArray().toOption.get.toVector, expected.map(_._1) ++ initial.drop(count).toVector)
              assertEquals(target.copyToArray().toOption.get.toVector, expected.map(_._2) ++ Vector.fill(32)(-999))
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("tuple flatMap retains outer Double and Boolean fields across inner generators and scalar folds"):
    val definition = ScalaKernels.tupleNested
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
              val expected = initial.take(count).map { original =>
                var total = original
                for outer <- 0 until rounds; inner <- 0 until outer if inner % 2 == 0 && enabled && inner + outer > 0
                    k <- 0 until 2 do
                  val item = if k == 0 then original / 2.0 else original / 2.0 + 1.0
                  total = total / 2.0 - item
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

  test("tuple Float Int folds retain ordered sums counts lazy reads and empty seeds on the GPU"):
    checkTupleFoldRows(ScalaKernels.tupleFoldRows)

  test("named Float Int folds retain ordered sums counts lazy reads and empty seeds on the GPU"):
    checkTupleFoldRows(ScalaKernels.namedFoldRows)

  private def checkTupleFoldRows(definition: Kernel[(DeviceBuffer[Float], DeviceBuffer[Float], DeviceBuffer[Int], Int, Int, Float)]): Unit =
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for rows <- counts; columns <- Vector(0, 1, 3, 7, 17) do
          val initial = Array.tabulate(math.max(1, rows * columns))(i => Vector(1.0e8f, 0.0f, 1.0f, -0.5f, -1.0e8f, 0.0f, 2.0f)(i % 7))
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](rows + 32).toOption.get
          val visits = context.allocate[Int](rows + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for seed <- Vector(0.0f, 7.0f, -3.0f); explicit <- Vector(false, true) do
              val expected = Vector.tabulate(rows) { row =>
                (0 until columns).filter(column => column % 2 == 0 && initial(row * columns + column) != 0.0f)
                  .foldLeft((seed, 0)) { (state, column) =>
                    (state._1 - initial(row * columns + column) - (if state._2 % 2 == 0 then 0.5f else 0.0f), state._2 + 1)
                  }
              }
              assertEquals(target.copyFrom(Array.fill(rows + 32)(-999f)), Right(()))
              assertEquals(visits.copyFrom(Array.fill(rows + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, visits, rows, columns, seed)), config(rows), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected.map(_._1) ++ Vector.fill(32)(-999f))
              assertEquals(visits.copyToArray().toOption.get.toVector, expected.map(_._2) ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            visits.close()
            target.close()
            data.close()
      finally stream.close()
    }

  test("tuple Int swaps and reused seeds preserve previous fields saved results and captured bounds"):
    checkTupleFoldReuse(ScalaKernels.tupleFoldReuse)

  test("named Int swaps and reused seeds preserve previous fields on the GPU"):
    checkTupleFoldReuse(ScalaKernels.namedFoldReuse)

  private def checkTupleFoldReuse(definition: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int, Int, Int)]): Unit =
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(count + 32)(i => i % 11 - 5)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](initial.length).toOption.get
          try
            for (from, until) <- Vector((0, 0), (3, 1), (-3, 2), (0, 1), (0, 7), (2, 9)); explicit <- Vector(false, true) do
              val expected = initial.take(count).map { original =>
                val indices = (from until until).filter(_ % 2 == 0)
                val first = indices.foldLeft((original, original + 1))((state, i) => (state._2, state._1 - (original + i)))
                val second = indices.foldLeft(first)((state, i) => (state._2 + (first._1 + i), state._1))
                (first._1 + second._2, first._2 + second._1)
              }.toVector
              assertEquals(data.copyFrom(initial), Right(()))
              assertEquals(target.copyFrom(Array.fill(initial.length)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, from, until)), config(count), stream, explicit)
              assertEquals(data.copyToArray().toOption.get.toVector, expected.map(_._1) ++ initial.drop(count).toVector)
              assertEquals(target.copyToArray().toOption.get.toVector, expected.map(_._2) ++ Vector.fill(32)(-999))
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("nested Double Boolean tuple folds refresh seeds and use previous state across flatMap"):
    checkTupleFoldNested(ScalaKernels.tupleFoldNested)

  test("nested named Double Boolean folds preserve previous state across GPU flatMap"):
    checkTupleFoldNested(ScalaKernels.namedFoldNested)

  private def checkTupleFoldNested(definition: Kernel[(DeviceBuffer[Double], DeviceBuffer[Double], Int, Int, Boolean)]): Unit =
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
              val expected = initial.take(count).map { original =>
                var total = original
                for round <- 0 until rounds do
                  val indices = for i <- 0 until round; j <- 0 until i if enabled yield j
                  val result = indices.foldLeft((total, false)) { (state, _) =>
                    (if state._2 then state._1 / 2.0 - original else state._1 - original, (state._1 > 0.0) != state._2)
                  }
                  total = result._1 + (if result._2 then 0.25 else 0.5)
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

  test("shared Float exchange crosses warp boundaries with partial blocks and guarded tight inputs"):
    checkSharedExchange(ScalaKernels.sharedExchange)

  test("block phase publishes Float tiles before cross-warp reads on both GPU stream paths"):
    checkSharedExchange(ScalaKernels.phaseExchange)

  private def checkSharedExchange(definition: Kernel[(DeviceBuffer[Float], DeviceBuffer[Float], Int)]): Unit =
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i - 90).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val neighbor = (i / 64) * 64 + (i % 64 + 1) % 64
              if neighbor < count then initial(neighbor) else 0.0f
            }
            assert(function.launch(definition.bind((data, target, count)), LaunchConfig(Grid.x(1), LaunchBlock.x(32))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("shared Int reuse preserves per-round snapshots before overwrite on both GPU stream paths"):
    checkSharedReuse(ScalaKernels.sharedReuse)

  test("repeated block phases retain Int snapshots before shared overwrite with empty and partial work"):
    checkSharedReuse(ScalaKernels.phaseReuse)

  private def checkSharedReuse(definition: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int, Int)]): Unit =
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 17 - 8)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rounds <- Vector(-1, 0, 1, 2, 33, 65); explicit <- Vector(false, true) do
              val expected = (0 until count by 64).flatMap { base =>
                var tile = Vector.tabulate(64)(lane => if base + lane < count then initial(base + lane) else 0)
                for round <- 0 until rounds do
                  tile = Vector.tabulate(64)(lane => tile((lane + 1) % 64) + round)
                tile.take(math.min(64, count - base))
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

  test("Boolean and Double shared storage handles disabled empty partial and multi-block GPU work"):
    val definition = ScalaKernels.sharedFlags
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i - 90).toDouble * 0.25)
          val data = context.allocate[Double](initial.length).toOption.get
          val target = context.allocate[Double](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for enabled <- Vector(false, true); explicit <- Vector(false, true) do
              val expected = Vector.tabulate(count) { i =>
                val neighbor = (i / 64) * 64 + (i % 64 + 1) % 64
                if enabled && neighbor < count then initial(neighbor) else -7.0
              }
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              launch(context, function, definition.bind((data, target, count, enabled)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999.0))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("rank-2 shared tiles exchange row-local columns with partial blocks and guarded tight inputs"):
    val definition = ScalaKernels.sharedTile
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i - 90).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val lane = i % 64
              val neighbor = (i / 64) * 64 + (lane / 32) * 32 + (lane % 32 + 1) % 32
              if neighbor < count then initial(neighbor) else 0.0f
            }
            assert(function.launch(definition.bind((data, target, count)), LaunchConfig(Grid.x(1), LaunchBlock.x(32))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("rank-3 shared cubes retain per-round cross-layer snapshots on both GPU stream paths"):
    val definition = ScalaKernels.sharedCube
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 17 - 8)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rounds <- Vector(-1, 0, 1, 2, 33, 65); explicit <- Vector(false, true) do
              val expected = (0 until count by 64).flatMap { base =>
                var cube = Vector.tabulate(2, 2, 16) { (layer, row, column) =>
                  val lane = layer * 32 + row * 16 + column
                  if base + lane < count then initial(base + lane) else 0
                }
                for round <- 0 until rounds do
                  cube = Vector.tabulate(2, 2, 16)((layer, row, column) => cube((layer + 1) % 2)(row)(column) + round)
                (0 until math.min(64, count - base)).map(lane => cube(lane / 32)((lane / 16) % 2)(lane % 16))
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

  test("dynamic shared exchange sizes storage at launch and rejects missing or misaligned bytes"):
    val definition = ScalaKernels.dynamicExchange
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i - 90).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val neighbor = (i / 64) * 64 + (i % 64 + 1) % 64
              if neighbor < count then initial(neighbor) else 0.0f
            }
            val dynamicConfig = config(count).copy(dynamicSharedMemoryBytes = 64 * 4)
            assert(function.launch(definition.bind((data, target, count)), config(count)).isLeft)
            assert(function.launch(definition.bind((data, target, count)),
              dynamicConfig.copy(dynamicSharedMemoryBytes = 63)).isLeft)
            assert(function.launch(definition.bind((data, target, count)),
              LaunchConfig(Grid.x(1), LaunchBlock.x(32), dynamicSharedMemoryBytes = 64 * 4)).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count)), dynamicConfig, stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  private def config32(count: Int): LaunchConfig =
    LaunchConfig(Grid.x(math.max(1, (count + 31) / 32)), LaunchBlock.x(32))

  test("warp tree maxima converge per full warp with partial blocks on both GPU stream paths"):
    val definition = ScalaKernels.warpTreeMax
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i - 90).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 32) * 32
              (0 until 32).map(lane => if base + lane < count then initial(base + lane) else -1000.0f).max
            }
            assert(function.launch(definition.bind((data, target, count)), LaunchConfig(Grid.x(1), LaunchBlock.x(64))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("noncommutative width-8 trees preserve exact subtree order on both GPU stream paths"):
    val definition = ScalaKernels.warpTreeDifference
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 17 - 8)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            def tree(group: Vector[Int]): Vector[Int] =
              Vector(1, 2, 4).foldLeft(group) { (current, distance) =>
                Vector.tabulate(8) { lane =>
                  val partner = current(lane ^ distance)
                  val left = if (lane & distance) == 0 then current(lane) else partner
                  val right = if (lane & distance) == 0 then partner else current(lane)
                  left - right
                }
              }
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 32) * 32 + ((i % 32) / 8) * 8
              val group = Vector.tabulate(8)(lane => if base + lane < count then initial(base + lane) else 0)
              tree(group)(i % 8)
            }
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("warp phases publish Float tiles to full-mask lanes before cross-lane reads on both GPU stream paths"):
    val definition = ScalaKernels.warpExchange
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i - 90).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val neighbor = (i / 32) * 32 + (i % 32 + 1) % 32
              if neighbor < count then initial(neighbor) else 0.0f
            }
            assert(function.launch(definition.bind((data, target, count)), LaunchConfig(Grid.x(1), LaunchBlock.x(64))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("repeated warp phases retain Int snapshots before shared overwrite with empty and partial work"):
    val definition = ScalaKernels.warpReuse
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 17 - 8)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rounds <- Vector(-1, 0, 1, 2, 33, 65); explicit <- Vector(false, true) do
              val expected = (0 until count by 32).flatMap { base =>
                var tile = Vector.tabulate(32)(lane => if base + lane < count then initial(base + lane) else 0)
                for round <- 0 until rounds do
                  tile = Vector.tabulate(32)(lane => tile((lane + 1) % 32) + round)
                tile.take(math.min(32, count - base))
              }.toVector
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count, rounds)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("disjoint warp groups exchange Int tiles independently under divergent lanes on the GPU"):
    val definition = ScalaKernels.warpGroups
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 19 - 9)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val neighbor = (i / 32) * 32 + (i % 32 + 2) % 32
              if neighbor < count then initial(neighbor) else -1
            }
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted warp sums reduce full 32-lane groups with zeroed tails on both GPU stream paths"):
    val definition = ScalaKernels.warpSum
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 9 - 4).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 32) * 32
              (0 until 32).map(lane => if base + lane < count then initial(base + lane) else 0.0f).sum
            }
            assert(function.launch(definition.bind((data, target, count)), LaunchConfig(Grid.x(1), LaunchBlock.x(64))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted half-warp Int group sums stay independent per aligned 16-lane group on the GPU"):
    val definition = ScalaKernels.warpGroupSums
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 23 - 11)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 16) * 16
              (0 until 16).map(lane => if base + lane < count then initial(base + lane) else 0).sum
            }
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("repeated quoted width-8 Double sums accumulate per round with sentinel tail lanes"):
    val definition = ScalaKernels.warpSumRounds
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 17 - 8).toDouble * 0.25)
          val data = context.allocate[Double](initial.length).toOption.get
          val target = context.allocate[Double](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rounds <- Vector(-1, 0, 1, 2, 5); explicit <- Vector(false, true) do
              val expected = Vector.tabulate(count) { i =>
                val base = (i / 8) * 8
                val group = (0 until 8).map(lane => if base + lane < count then initial(base + lane) else -0.5).sum
                group * math.max(0, rounds)
              }
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              launch(context, function, definition.bind((data, target, count, rounds)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999.0))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted block sums reduce whole 256-thread blocks with zeroed tails on both GPU stream paths"):
    val definition = ScalaKernels.blockSum
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 9 - 4).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 256) * 256
              (0 until 256).map(lane => if base + lane < count then initial(base + lane) else 0.0f).sum
            }
            assert(function.launch(definition.bind((data, target, count)),
              LaunchConfig(Grid.x(1), LaunchBlock.x(64))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count)),
                LaunchConfig(Grid.x(math.max(1, (count + 255) / 256)), LaunchBlock.x(256)), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted block maxima combine whole 64-thread blocks with sentinel tail items on the GPU"):
    val definition = ScalaKernels.blockTreeMax
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 23 - 11)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 64) * 64
              (0 until 64).map(lane => if base + lane < count then initial(base + lane) else -1000000).max
            }
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count)),
                LaunchConfig(Grid.x(math.max(1, (count + 63) / 64)), LaunchBlock.x(64)), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("reused quoted block scratch keeps Double sums and minima separate per 128-thread block"):
    val definition = ScalaKernels.blockSumReuse
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 17 - 8).toDouble * 0.25)
          val data = context.allocate[Double](initial.length).toOption.get
          val target = context.allocate[Double](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 128) * 128
              val items = (0 until 128).map(lane => if base + lane < count then initial(base + lane) else 0.25)
              items.sum - items.min
            }
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              launch(context, function, definition.bind((data, target, count)),
                LaunchConfig(Grid.x(math.max(1, (count + 127) / 128)), LaunchBlock.x(128)), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999.0))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted direct shuffles broadcast and rotate Int lanes within each warp on both GPU stream paths"):
    val definition = ScalaKernels.shuffleRotate
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 23 - 11)
          val data = context.allocate[Int](initial.length).toOption.get
          val target = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            def item(index: Int): Int = if index < count then initial(index) else -7
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 32) * 32
              item(base) + item(base + (i % 32 + 1) % 32)
            }
            assert(function.launch(definition.bind((data, target, count)), LaunchConfig(Grid.x(1), LaunchBlock.x(64))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted shuffle-up scans compute exact inclusive Float prefix sums per warp"):
    val definition = ScalaKernels.shufflePrefix
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 9 - 4).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 32) * 32
              (base to i).map(k => if k < count then initial(k) else 0.0f).sum
            }
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted width-8 down and xor shuffles exchange Double segments with sentinel tails"):
    val definition = ScalaKernels.shuffleFold
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 17 - 8).toDouble * 0.25)
          val data = context.allocate[Double](initial.length).toOption.get
          val target = context.allocate[Double](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            def item(index: Int): Double = if index < count then initial(index) else 0.5
            val expected = Vector.tabulate(count) { i =>
              val segment = (i / 8) * 8
              val j = i % 8
              val down = if j + 2 < 8 then item(segment + j + 2) else item(i)
              val partner = item(segment + (j ^ 1))
              down - partner
            }
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config32(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999.0))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted atomic adds count Int histogram bins race-free across blocks on both GPU stream paths"):
    val definition = ScalaKernels.atomicHistogram
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 20 - 7)
          val data = context.allocate[Int](initial.length).toOption.get
          val bins = context.allocate[Int](8).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(8) { bin =>
              bin * 100 + initial.take(count).count(value => ((value % 8) + 8) % 8 == bin)
            }
            for explicit <- Vector(false, true) do
              assertEquals(bins.copyFrom(Array.tabulate(8)(bin => bin * 100)), Right(()))
              launch(context, function, definition.bind((data, bins, count)), config(count), stream, explicit)
              assertEquals(bins.copyToArray().toOption.get.toVector, expected)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            bins.close()
            data.close()
      finally stream.close()
    }

  test("quoted atomic fetch adds reserve unique compaction slots holding every positive Float"):
    val definition = ScalaKernels.atomicCompact
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 9 - 4).toFloat * 0.25f)
          val positives = initial.take(count).filter(_ > 0.0f)
          val data = context.allocate[Float](initial.length).toOption.get
          val packed = context.allocate[Float](count + 32).toOption.get
          val cursor = context.allocate[Int](1).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for explicit <- Vector(false, true) do
              assertEquals(packed.copyFrom(Array.fill(count + 32)(-999f)), Right(()))
              assertEquals(cursor.copyFrom(Array(0)), Right(()))
              launch(context, function, definition.bind((data, packed, cursor, count)), config(count), stream, explicit)
              assertEquals(cursor.copyToArray().toOption.get.toVector, Vector(positives.length))
              val packedOut = packed.copyToArray().toOption.get
              // Slot order is unspecified; the packed prefix must be a permutation of the positives.
              assertEquals(packedOut.take(positives.length).sorted.toVector, positives.sorted.toVector)
              assert(packedOut.drop(positives.length).forall(_ == -999f))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            cursor.close()
            packed.close()
            data.close()
      finally stream.close()
    }

  test("quoted shared atomic adds accumulate exact per-block Float totals with zeroed tails"):
    val definition = ScalaKernels.atomicSharedSum
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val blocks = math.max(1, (count + 63) / 64)
          val initial = Array.tabulate(math.max(1, count))(i => (i % 9 - 4).toFloat * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val totals = context.allocate[Float](blocks).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(blocks) { block =>
              (0 until 64).map(lane => if block * 64 + lane < count then initial(block * 64 + lane) else 0.0f).sum
            }
            assert(function.launch(definition.bind((data, totals, count)), LaunchConfig(Grid.x(1), LaunchBlock.x(32))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(totals.copyFrom(Array.fill(blocks)(-999f)), Right(()))
              launch(context, function, definition.bind((data, totals, count)), config(count), stream, explicit)
              assertEquals(totals.copyToArray().toOption.get.toVector, expected)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            totals.close()
            data.close()
      finally stream.close()
    }

  test("quoted scoped release stores publish payloads that acquire loads observe across every block"):
    val definition = ScalaKernels.scopedFlag
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val blocks = math.max(1, (count + 63) / 64)
          val initial = Array.tabulate(math.max(1, count))(i => i % 19 - 9)
          val data = context.allocate[Int](initial.length).toOption.get
          val cell = context.allocate[Int](blocks).toOption.get
          val flag = context.allocate[Int](blocks).toOption.get
          val out = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val base = (i / 64) * 64
              if base < count then initial(base) else -5
            }
            for explicit <- Vector(false, true) do
              assertEquals(cell.copyFrom(Array.fill(blocks)(-111)), Right(()))
              assertEquals(flag.copyFrom(Array.fill(blocks)(0)), Right(()))
              assertEquals(out.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, cell, flag, out, count)), config(count), stream, explicit)
              assertEquals(out.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
              assertEquals(flag.copyToArray().toOption.get.toVector, Vector.fill(blocks)(1))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            out.close()
            flag.close()
            cell.close()
            data.close()
      finally stream.close()
    }

  test("quoted scoped exchanges and fetch subtractions return exact old values per element"):
    val definition = ScalaKernels.scopedExchange
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 19 - 9)
          val seeds = Array.tabulate(math.max(1, count))(i => i % 13 - 6)
          val data = context.allocate[Int](initial.length).toOption.get
          val cells = context.allocate[Int](seeds.length).toOption.get
          val out = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expectedOut = Vector.tabulate(count)(i => seeds(i) + initial(i))
            val expectedCells = Vector.tabulate(seeds.length)(i => if i < count then initial(i) - seeds(i) else seeds(i))
            for explicit <- Vector(false, true) do
              assertEquals(cells.copyFrom(seeds), Right(()))
              assertEquals(out.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, cells, out, count)), config(count), stream, explicit)
              assertEquals(out.copyToArray().toOption.get.toVector, expectedOut ++ Vector.fill(32)(-999))
              assertEquals(cells.copyToArray().toOption.get.toVector, expectedCells)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            out.close()
            cells.close()
            data.close()
      finally stream.close()
    }

  test("quoted block-scope shared counters accumulate exact Double totals through scoped atomics"):
    val definition = ScalaKernels.scopedSharedCounter
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val blocks = math.max(1, (count + 63) / 64)
          val initial = Array.tabulate(math.max(1, count))(i => (i % 17 - 8).toDouble * 0.25)
          val data = context.allocate[Double](initial.length).toOption.get
          val totals = context.allocate[Double](blocks).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(blocks) { block =>
              (0 until 64).map(lane => if block * 64 + lane < count then initial(block * 64 + lane) else 0.25).sum
            }
            assert(function.launch(definition.bind((data, totals, count)), LaunchConfig(Grid.x(1), LaunchBlock.x(32))).isLeft)
            for explicit <- Vector(false, true) do
              assertEquals(totals.copyFrom(Array.fill(blocks)(-999.0)), Right(()))
              launch(context, function, definition.bind((data, totals, count)), config(count), stream, explicit)
              assertEquals(totals.copyToArray().toOption.get.toVector, expected)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            totals.close()
            data.close()
      finally stream.close()
    }

  test("quoted scoped minima and maxima return exact old values and clamp every element"):
    val definition = ScalaKernels.scopedBounds
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 19 - 9)
          val floorSeeds = Array.tabulate(math.max(1, count))(i => i % 13 - 6)
          val ceilingSeeds = Array.tabulate(math.max(1, count))(i => i % 7 - 3)
          val data = context.allocate[Int](initial.length).toOption.get
          val floors = context.allocate[Int](floorSeeds.length).toOption.get
          val ceilings = context.allocate[Int](ceilingSeeds.length).toOption.get
          val out = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expectedFloors = Vector.tabulate(floorSeeds.length)(i =>
              if i < count then math.min(floorSeeds(i), initial(i)) else floorSeeds(i))
            val expectedCeilings = Vector.tabulate(ceilingSeeds.length)(i =>
              if i < count then math.max(ceilingSeeds(i), initial(i)) else ceilingSeeds(i))
            val expectedOut = Vector.tabulate(count)(i => floorSeeds(i) + ceilingSeeds(i))
            for explicit <- Vector(false, true) do
              assertEquals(floors.copyFrom(floorSeeds), Right(()))
              assertEquals(ceilings.copyFrom(ceilingSeeds), Right(()))
              assertEquals(out.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, floors, ceilings, out, count)), config(count), stream, explicit)
              assertEquals(out.copyToArray().toOption.get.toVector, expectedOut ++ Vector.fill(32)(-999))
              assertEquals(floors.copyToArray().toOption.get.toVector, expectedFloors)
              assertEquals(ceilings.copyToArray().toOption.get.toVector, expectedCeilings)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            out.close()
            ceilings.close()
            floors.close()
            data.close()
      finally stream.close()
    }

  test("quoted scoped bitwise fetches thread old values through chained masks per element"):
    val definition = ScalaKernels.scopedBitmask
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i % 19 - 9)
          val maskSeeds = Array.tabulate(math.max(1, count))(i => i % 23 - 11)
          val data = context.allocate[Int](initial.length).toOption.get
          val masks = context.allocate[Int](maskSeeds.length).toOption.get
          val andOld = context.allocate[Int](count + 32).toOption.get
          val orOld = context.allocate[Int](count + 32).toOption.get
          val xorOld = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val afterAnd = Vector.tabulate(count)(i => maskSeeds(i) & initial(i))
            val afterOr = afterAnd.map(_ | 5)
            val expectedMasks = Vector.tabulate(maskSeeds.length)(i =>
              if i < count then afterOr(i) ^ 3 else maskSeeds(i))
            for explicit <- Vector(false, true) do
              assertEquals(masks.copyFrom(maskSeeds), Right(()))
              for buffer <- Vector(andOld, orOld, xorOld) do
                assertEquals(buffer.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, masks, andOld, orOld, xorOld, count)), config(count), stream, explicit)
              assertEquals(andOld.copyToArray().toOption.get.toVector,
                Vector.tabulate(count)(maskSeeds(_)) ++ Vector.fill(32)(-999))
              assertEquals(orOld.copyToArray().toOption.get.toVector, afterAnd ++ Vector.fill(32)(-999))
              assertEquals(xorOld.copyToArray().toOption.get.toVector, afterOr ++ Vector.fill(32)(-999))
              assertEquals(masks.copyToArray().toOption.get.toVector, expectedMasks)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            xorOld.close()
            orOld.close()
            andOld.close()
            masks.close()
            data.close()
      finally stream.close()
    }

  test("quoted compare exchanges claim one winner per block and swap exactly the matching elements"):
    val definition = ScalaKernels.scopedClaim
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val blocks = math.max(1, (count + 63) / 64)
          val initial = Array.tabulate(math.max(1, count))(i => i % 19 - 9)
          val cellSeeds = Array.tabulate(math.max(1, count))(i => i % 13 - 6)
          val data = context.allocate[Int](initial.length).toOption.get
          val cells = context.allocate[Int](cellSeeds.length).toOption.get
          val winners = context.allocate[Int](blocks).toOption.get
          val out = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expectedCells = Vector.tabulate(cellSeeds.length)(i =>
              if i < count && i % 2 == 0 then initial(i) else cellSeeds(i))
            val expectedOut = Vector.tabulate(count)(cellSeeds(_)) ++
              Vector.fill(31)(-999) :+ 0
            for explicit <- Vector(false, true) do
              assertEquals(cells.copyFrom(cellSeeds), Right(()))
              assertEquals(winners.copyFrom(Array.fill(blocks)(0)), Right(()))
              assertEquals(out.copyFrom(Array.fill(count + 32)(-999)), Right(()))
              launch(context, function, definition.bind((data, cells, winners, out, count)), config(count), stream, explicit)
              assertEquals(out.copyToArray().toOption.get.toVector, expectedOut)
              assertEquals(cells.copyToArray().toOption.get.toVector, expectedCells)
              assertEquals(winners.copyToArray().toOption.get.toVector, Vector.fill(blocks)(1))
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            out.close()
            winners.close()
            cells.close()
            data.close()
      finally stream.close()
    }

  test("quoted activation math matches JVM references within documented libm tolerances"):
    val definition = ScalaKernels.mathActivation
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 19 - 9) * 0.25f)
          val mean = 0.5f
          val variance = 2.0f
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count)(i =>
              math.tanh((initial(i) - mean) / math.sqrt(variance + 0.00001f)))
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999.0f)), Right(()))
              launch(context, function, definition.bind((data, target, mean, variance, count)), config(count), stream, explicit)
              val produced = target.copyToArray().toOption.get
              for i <- 0 until count do
                assert(math.abs(produced(i) - expected(i)) <= 1e-4f, s"$i: ${produced(i)} vs ${expected(i)}")
              for i <- count until count + 32 do assertEquals(produced(i), -999.0f)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            target.close()
            data.close()
      finally stream.close()
    }

  test("quoted logistic math matches JVM exp and log references in Double precision"):
    val definition = ScalaKernels.mathLogistic
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 19 - 9) * 0.5)
          val data = context.allocate[Double](initial.length).toOption.get
          val sigmoid = context.allocate[Double](count + 32).toOption.get
          val logs = context.allocate[Double](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expectedSigmoid = Vector.tabulate(count)(i => 1.0 / (1.0 + math.exp(0.0 - initial(i))))
            val expectedLogs = Vector.tabulate(count)(i => math.log(initial(i) * initial(i) + 1.0))
            for explicit <- Vector(false, true) do
              assertEquals(sigmoid.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              assertEquals(logs.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              launch(context, function, definition.bind((data, sigmoid, logs, count)), config(count), stream, explicit)
              val producedSigmoid = sigmoid.copyToArray().toOption.get
              val producedLogs = logs.copyToArray().toOption.get
              for i <- 0 until count do
                assert(math.abs(producedSigmoid(i) - expectedSigmoid(i)) <= 1e-12, s"sigmoid $i: ${producedSigmoid(i)} vs ${expectedSigmoid(i)}")
                assert(math.abs(producedLogs(i) - expectedLogs(i)) <= 1e-12, s"log $i: ${producedLogs(i)} vs ${expectedLogs(i)}")
              for i <- count until count + 32 do
                assertEquals(producedSigmoid(i), -999.0)
                assertEquals(producedLogs(i), -999.0)
              assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            logs.close()
            sigmoid.close()
            data.close()
      finally stream.close()
    }

  test("quoted distance math branches on sqrt results and logs the short radii"):
    val definition = ScalaKernels.mathDistance
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val xs = Array.tabulate(math.max(1, count))(i => (i % 19 - 9) * 0.5f)
          val ys = Array.tabulate(math.max(1, count))(i => (i % 13 - 6) * 0.5f)
          val cutoff = 3.0f
          val xBuffer = context.allocate[Float](xs.length).toOption.get
          val yBuffer = context.allocate[Float](ys.length).toOption.get
          val out = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(xBuffer.copyFrom(xs), Right(()))
            assertEquals(yBuffer.copyFrom(ys), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val radius = math.sqrt(xs(i).toDouble * xs(i) + ys(i).toDouble * ys(i))
              if radius > cutoff then radius else math.log(radius + 1.0)
            }
            for explicit <- Vector(false, true) do
              assertEquals(out.copyFrom(Array.fill(count + 32)(-999.0f)), Right(()))
              launch(context, function, definition.bind((xBuffer, yBuffer, out, cutoff, count)), config(count), stream, explicit)
              val produced = out.copyToArray().toOption.get
              for i <- 0 until count do
                assert(math.abs(produced(i) - expected(i)) <= 1e-4f, s"$i: ${produced(i)} vs ${expected(i)}")
              for i <- count until count + 32 do assertEquals(produced(i), -999.0f)
              assertEquals(xBuffer.copyToArray().toOption.get.toVector, xs.toVector)
              assertEquals(yBuffer.copyToArray().toOption.get.toVector, ys.toVector)
          finally
            out.close()
            yBuffer.close()
            xBuffer.close()
      finally stream.close()
    }

  test("quoted shifts rotate pack and complement exactly like JVM Int semantics"):
    val definition = ScalaKernels.bitPack
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => i * 2654435761L.toInt + (i % 19 - 9))
          val data = context.allocate[Int](initial.length).toOption.get
          val packed = context.allocate[Int](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            for rotate <- Vector(0, 7, 31) do
              val expected = Vector.tabulate(count) { i =>
                val x = initial(i)
                ((x << rotate) | (x >>> (32 - rotate))) ^ ~(x >> 2)
              }
              for explicit <- Vector(false, true) do
                assertEquals(packed.copyFrom(Array.fill(count + 32)(-999)), Right(()))
                launch(context, function, definition.bind((data, packed, rotate, count)), config(count), stream, explicit)
                assertEquals(packed.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999))
                assertEquals(data.copyToArray().toOption.get.toVector, initial.toVector)
          finally
            packed.close()
            data.close()
      finally stream.close()
    }

  test("quoted Float and Double negations flip signs exactly including zeros"):
    val definition = ScalaKernels.negateBlend
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val floats = Array.tabulate(math.max(1, count))(i => (i % 19 - 9) * 0.25f)
          val doubles = Array.tabulate(math.max(1, count))(i => (i % 13 - 6) * 0.5)
          val weight = 1.5f
          val floatBuffer = context.allocate[Float](floats.length).toOption.get
          val outF = context.allocate[Float](count + 32).toOption.get
          val doubleBuffer = context.allocate[Double](doubles.length).toOption.get
          val outD = context.allocate[Double](count + 32).toOption.get
          try
            assertEquals(floatBuffer.copyFrom(floats), Right(()))
            assertEquals(doubleBuffer.copyFrom(doubles), Right(()))
            val expectedF = Vector.tabulate(count)(i => -floats(i) * weight + -(floats(i) - weight))
            val expectedD = Vector.tabulate(count)(i => -doubles(i) * 0.5 + -(doubles(i) * doubles(i)))
            for explicit <- Vector(false, true) do
              assertEquals(outF.copyFrom(Array.fill(count + 32)(-999.0f)), Right(()))
              assertEquals(outD.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              launch(context, function, definition.bind((floatBuffer, outF, doubleBuffer, outD, weight, count)), config(count), stream, explicit)
              assertEquals(outF.copyToArray().toOption.get.toVector, expectedF ++ Vector.fill(32)(-999.0f))
              assertEquals(outD.copyToArray().toOption.get.toVector, expectedD ++ Vector.fill(32)(-999.0))
              assertEquals(floatBuffer.copyToArray().toOption.get.toVector, floats.toVector)
              assertEquals(doubleBuffer.copyToArray().toOption.get.toVector, doubles.toVector)
          finally
            outD.close()
            doubleBuffer.close()
            outF.close()
            floatBuffer.close()
      finally stream.close()
    }

  test("quoted conversions reproduce JVM widening and nearest-even narrowing bit-for-bit"):
    val definition = ScalaKernels.convertBlend
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val ints = Array.tabulate(math.max(1, count))(i => i * 2654435761L.toInt + (i % 19 - 9))
          val doubles = Array.tabulate(math.max(1, count))(i => (i % 23 - 11) * 0.37 + 1.0 / 3.0)
          val intBuffer = context.allocate[Int](ints.length).toOption.get
          val doubleBuffer = context.allocate[Double](doubles.length).toOption.get
          val outF = context.allocate[Float](count + 32).toOption.get
          val outD = context.allocate[Double](count + 32).toOption.get
          try
            assertEquals(intBuffer.copyFrom(ints), Right(()))
            assertEquals(doubleBuffer.copyFrom(doubles), Right(()))
            val expectedF = Vector.tabulate(count)(i => ints(i).toFloat * 0.5f + doubles(i).toFloat)
            val expectedD = Vector.tabulate(count)(i => ints(i).toDouble + doubles(i) + doubles(i).toFloat.toDouble)
            for explicit <- Vector(false, true) do
              assertEquals(outF.copyFrom(Array.fill(count + 32)(-999.0f)), Right(()))
              assertEquals(outD.copyFrom(Array.fill(count + 32)(-999.0)), Right(()))
              launch(context, function, definition.bind((intBuffer, doubleBuffer, outF, outD, count)), config(count), stream, explicit)
              assertEquals(outF.copyToArray().toOption.get.toVector, expectedF ++ Vector.fill(32)(-999.0f))
              assertEquals(outD.copyToArray().toOption.get.toVector, expectedD ++ Vector.fill(32)(-999.0))
              assertEquals(intBuffer.copyToArray().toOption.get.toVector, ints.toVector)
              assertEquals(doubleBuffer.copyToArray().toOption.get.toVector, doubles.toVector)
          finally
            outD.close()
            outF.close()
            doubleBuffer.close()
            intBuffer.close()
      finally stream.close()
    }

  test("quoted local arrays keep per-thread windows exact across static and dynamic indices"):
    val definition = ScalaKernels.localWindow
    withKernel(definition) { (context, function) =>
      val stream = context.createStream().toOption.get
      try
        for count <- counts do
          val initial = Array.tabulate(math.max(1, count))(i => (i % 19 - 9) * 0.25f)
          val data = context.allocate[Float](initial.length).toOption.get
          val target = context.allocate[Float](count + 32).toOption.get
          try
            assertEquals(data.copyFrom(initial), Right(()))
            val expected = Vector.tabulate(count) { i =>
              val window = new Array[Float](4)
              window(0) = initial(i)
              window(1) = window(0) * 0.5f
              window(2) = window(1) + 1.0f
              val pick = (i % 2) + 1
              window(3) = window(pick)
              val slot = i % 3
              window(slot) = window(slot) + 2.0f
              window(0) + window(1) + window(2) + window(3)
            }
            for explicit <- Vector(false, true) do
              assertEquals(target.copyFrom(Array.fill(count + 32)(-999.0f)), Right(()))
              launch(context, function, definition.bind((data, target, count)), config(count), stream, explicit)
              assertEquals(target.copyToArray().toOption.get.toVector, expected ++ Vector.fill(32)(-999.0f))
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
