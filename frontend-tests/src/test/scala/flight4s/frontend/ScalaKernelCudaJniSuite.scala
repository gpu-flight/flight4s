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
    val definition = ScalaKernels.sharedExchange
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
    val definition = ScalaKernels.sharedReuse
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
