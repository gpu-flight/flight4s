package flight4s.frontend.examples

import scala.annotation.experimental
import flight4s.frontend.ScalaKernel.*

@experimental
object ScalaKernels:
  def tupleScale = kernel("quotedTupleScale", params((input[Float]("data"), output[Float]("target"),
      value[Int]("count"), value[Float]("factor"), value[Float]("bias"),
      value[Boolean]("enabled"), value[Double]("cutoff")))) { (data, target, count, factor, bias, enabled, cutoff) =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < count && enabled && cutoff > 0.0 then
      target(i) = data(i) * factor + bias
  }

  def scale = kernel("quotedScale", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"), value[Float]("factor"))) { (data, target, count, factor) =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < count then
      val original = data(i)
      target(i) = original * factor
  }

  def branches = kernel("quotedBranches", params(inOut[Int]("data"), output[Int]("saved"),
      output[Int]("target"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._4 then
      var total = p._1(i)
      val before = total
      if i % 2 == 0 then total += 3 else total = total - 2
      p._1(i) = total
      p._2(i) = before
      if before < 0 then
        val total = before + 7
        p._3(i) = total
      else
        p._3(i) = if total > 0 then total else before
  }

  def shortCircuit = kernel("quotedShortCircuit", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._3 && (p._1(i) != 0 || p._1(i) == 0) then
      val nonzero = p._1(i) != 0
      p._2(i) = if !nonzero then 11 else p._1(i) + 2
  }

  def doubles = kernel("quotedDouble", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"), value[Boolean]("enabled"), value[Double]("bias"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if p._4 && i < p._3 then
      val old = p._1(i)
      var total = old
      total += p._5
      p._2(i) = if old < 0.0 then total * 2.0 else total / 2.0
  }

  def rowSum = kernel("quotedRowSum", params(input[Float]("data"), output[Float]("target"),
      value[Int]("rows"), value[Int]("columns"))) { (data, target, rows, columns) =>
    val row = blockIdx.x * blockDim.x + threadIdx.x
    if row < rows then
      var total = 0.0f
      for column <- 0 until columns do
        val item = data(row * columns + column)
        total += item
      target(row) = total
  }

  def rangeBounds = kernel("quotedRangeBounds", params(output[Int]("visits"), output[Int]("last"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var begin = p._4
      var end = p._5
      var visits = 0
      var last = 123
      for index <- begin until end do
        val snapshot = index
        visits += 1
        last = snapshot
        begin = 0
        end = 0
      p._1(lane) = visits
      p._2(lane) = last
  }

  def nestedRanges = kernel("quotedNestedRanges", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("rounds"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var total = p._1(lane)
      for index <- 0 until p._4 do
        val before = total
        for index <- 0 until index do
          val previous = total
          total = previous + before + index
      p._2(lane) = total
  }

  def guardedRows = kernel("quotedGuardedRows", params(input[Float]("data"), output[Float]("target"),
      value[Int]("rows"), value[Int]("columns"), value[Float]("threshold"))) { p =>
    val row = blockIdx.x * blockDim.x + threadIdx.x
    if row < p._3 then
      var total = 0.0f
      for column <- -1 until p._4 + 1 if column >= 0 if column < p._4 if p._1(row * p._4 + column) > p._5 do
        val item = p._1(row * p._4 + column)
        total += item
      p._2(row) = total
  }

  def guardedState = kernel("quotedGuardedState", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var total = p._1(lane)
      var begin = p._4
      var end = p._5
      (begin until end).withFilter(first => first >= total).withFilter(second => second % 2 == 0)
        .foreach { third =>
          val before = total
          total = before + third + 1
          begin = 0
          end = 0
        }
      p._2(lane) = total
  }

  def nestedGuards = kernel("quotedNestedGuards", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("rounds"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var total = p._1(lane)
      for outer <- 0 until p._4 if outer % 2 == 0; inner <- -1 until outer + 1 if inner >= 0 if inner < outer do
        val before = total
        total = before + outer + inner
      for index <- 0 until p._4 if index > 0 do
        for index <- 0 until index if index % 2 == 0 do
          total += index
      p._2(lane) = total
  }

  def yieldRows = kernel("quotedYieldRows", params(input[Float]("data"), output[Float]("target"),
      value[Int]("rows"), value[Int]("columns"), value[Float]("threshold"))) { p =>
    val row = blockIdx.x * blockDim.x + threadIdx.x
    if row < p._3 then
      var total = 0.0f
      val values = for column <- deviceRange(-1, p._4 + 1) if column >= 0 if column < p._4
        yield p._1(row * p._4 + column)
      values.withFilter(item => item > p._5).map(item => item * 2.0f).foreach { item => total += item }
      p._2(row) = total
  }

  def yieldReuse = kernel("quotedYieldReuse", params(inOut[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var begin = p._4
      var end = p._5
      var bias = 0
      var total = 0
      val values = for index <- deviceRange(begin, end) if index % 2 == 0
        yield p._1(lane) + index + bias
      val alias = values
      begin = 0
      end = 0
      bias = 1
      alias.foreach { item =>
        p._1(lane) = item + 1
        total += item + item
      }
      bias = 3
      for item <- values do
        p._1(lane) = item + 1
        total += item + item
      p._2(lane) = total
  }

  def foldRows = kernel("quotedFoldRows", params(input[Float]("data"), output[Float]("target"),
      value[Int]("rows"), value[Int]("columns"), value[Float]("threshold"), value[Float]("seed"))) { p =>
    val row = blockIdx.x * blockDim.x + threadIdx.x
    if row < p._3 then
      val values = for column <- deviceRange(-1, p._4 + 1) if column >= 0 if column < p._4
        yield p._1(row * p._4 + column)
      val total = values.withFilter(item => item > p._5).map(item => item * 2.0f)
        .foldLeft(p._6)((sum, item) => sum - item)
      p._2(row) = total
  }

  def foldReuse = kernel("quotedFoldReuse", params(inOut[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var begin = p._4
      var end = p._5
      var bias = 0
      val values = for index <- deviceRange(begin, end) if index % 2 == 0
        yield p._1(lane) + index + bias
      val alias = values
      begin = 0
      end = 0
      bias = 1
      val first = alias.foldLeft(p._1(lane))((sum, item) => sum - item - item)
      p._1(lane) = first
      bias = 3
      var second = values.foldLeft(first + 1)((sum, item) => if item > sum then item else sum - item)
      second += first
      p._2(lane) = second
  }

  def foldNested = kernel("quotedFoldNested", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var total = p._1(lane)
      for index <- 0 until p._4 do
        val before = total
        val values = deviceRange(0, index).map(index => index + 1)
        val accepted = values.foldLeft(p._5)((accepted, item) => accepted || item % 2 == 0)
        var next = values.foldLeft(before)((before, item) => if accepted then before / 2.0 else before - 1.0)
        next += 0.25
        total = next
      p._2(lane) = total
  }

  def flatMapRows = kernel("quotedFlatMapRows", params(input[Float]("data"), input[Int]("lengths"),
      output[Float]("target"), value[Int]("rows"), value[Int]("columns"),
      value[Float]("threshold"))) { p =>
    val row = blockIdx.x * blockDim.x + threadIdx.x
    if row < p._4 then
      val values = for column <- deviceRange(-1, p._5 + 1) if column >= 0 if column < p._5
        inner <- deviceRange(0, p._2(column)) if inner % 2 == 0
      yield p._1(row * p._5 + column) + (if inner == 0 then 0.0f else 0.5f)
      val total = values.withFilter(item => item > p._6).map(item => item * 2.0f)
        .foldLeft(7.0f)((sum, item) => sum - item)
      p._3(row) = total
  }

  def flatMapReuse = kernel("quotedFlatMapReuse", params(inOut[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var begin = p._4
      var end = p._5
      var limit = 3
      var total = 0
      val values = deviceRange(begin, end).withFilter(outer => outer % 2 == 0)
        .map(outer => p._1(lane) + outer)
        .flatMap(outer => deviceRange(0, limit).map(inner => outer + inner))
        .withFilter(item => item >= 0)
      val alias = values
      begin = 0
      end = 0
      alias.foreach { item =>
        p._1(lane) = item + 1
        total += item + item
        limit = 1
      }
      limit = 2
      val result = values.foldLeft(total)((sum, item) => sum - item)
      p._2(lane) = result
  }

  def flatMapNested = kernel("quotedFlatMapNested", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      val base = deviceRange(0, p._4)
      val fixed = deviceRange(0, 2)
      val values = base.flatMap(outer => fixed.map(inner => outer + inner))
        .flatMap(index => deviceRange(0, index).map(index => index + 1))
        .map(item => if item % 2 == 0 then p._5 else !p._5)
      val found = values.foldLeft(false)((found, item) => found || item)
      val total = values.map(flag => if flag then 2.0 else 1.0)
        .foldLeft(p._1(lane))((sum, item) => sum / 2.0 - item)
      p._2(lane) = if found then total else p._1(lane)
  }

  def yieldNested = kernel("quotedYieldNested", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("rounds"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var total = p._1(lane)
      for outer <- 0 until p._4 do
        val before = total
        val values = deviceRange(0, outer).map(index => index + before)
          .withFilter(item => item >= total).map(item => item + outer)
        for before <- values do total += before
      p._2(lane) = total
  }
