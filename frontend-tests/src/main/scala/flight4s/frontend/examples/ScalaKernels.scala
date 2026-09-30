package flight4s.frontend.examples

import scala.annotation.experimental
import flight4s.frontend.ScalaKernel.*

@experimental
object ScalaKernels:
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
