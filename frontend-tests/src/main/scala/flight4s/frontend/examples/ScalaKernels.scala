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
