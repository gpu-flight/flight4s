package flight4s.frontend.examples

import scala.annotation.experimental
import flight4s.frontend.ScalaKernel.*
import flight4s.core.ir.MemoryOrder
import flight4s.core.launch.{Block as LaunchBlock}

@experimental
object ScalaKernels:
  def phaseExchange = kernel("quotedPhaseExchange", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { (data, target, count) =>
    val tile = sharedArray[Float](64)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    sync.blockAfter {
      val item = if i < count then data(i) else 0.0f
      tile(lane) = item
    }
    if i < count then target(i) = tile((lane + 1) % 64)
  }.requiringBlock(LaunchBlock.x(64))

  def phaseReuse = kernel("quotedPhaseReuse", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("rounds"))) { p =>
    val tile = sharedArray[Int](64)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    sync.blockAfter { tile(lane) = if i < p._3 then p._1(i) else 0 }
    for round <- deviceRange(0, p._4) do
      sync.blockAfter {
        val previous = tile((lane + 1) % 64)
        sync.block()
        tile(lane) = previous + round
      }
    if i < p._3 then p._2(i) = tile(lane)
  }.requiringBlock(LaunchBlock.x(64))

  def warpExchange = kernel("quotedWarpExchange", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { (data, target, count) =>
    val tile = sharedArray[Float](32)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    sync.warpAfter(sync.fullMask) {
      val item = if i < count then data(i) else 0.0f
      tile(lane) = item
    }
    if i < count then target(i) = tile((lane + 1) % 32)
  }.requiringBlock(LaunchBlock.x(32))

  def warpReuse = kernel("quotedWarpReuse", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("rounds"))) { p =>
    val tile = sharedArray[Int](32)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    sync.warpAfter(sync.fullMask) { tile(lane) = if i < p._3 then p._1(i) else 0 }
    for round <- deviceRange(0, p._4) do
      sync.warpAfter(sync.fullMask) {
        val previous = tile((lane + 1) % 32)
        sync.warp(sync.fullMask)
        tile(lane) = previous + round
      }
    if i < p._3 then p._2(i) = tile(lane)
  }.requiringBlock(LaunchBlock.x(32))

  def warpGroups = kernel("quotedWarpGroups", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"))) { p =>
    val tile = sharedArray[Int](32)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    tile(lane) = if i < p._3 then p._1(i) else -1
    if lane % 2 == 0 then
      sync.warp(0x55555555)
      if i < p._3 then p._2(i) = tile((lane + 2) % 32)
    else
      sync.warp(0xaaaaaaaa)
      if i < p._3 then p._2(i) = tile((lane + 2) % 32)
  }.requiringBlock(LaunchBlock.x(32))

  def warpSum = kernel("quotedWarpSum", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { (data, target, count) =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    val item = if i < count then data(i) else 0.0f
    val total = warpReduceSum(sync.fullMask, item)
    if i < count then target(i) = total
  }.requiringBlock(LaunchBlock.x(32))

  def warpGroupSums = kernel("quotedWarpGroupSums", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"))) { p =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    val item = if i < p._3 then p._1(i) else 0
    val total = warpReduceSum(sync.fullMask, item, 16)
    if i < p._3 then p._2(i) = total
  }.requiringBlock(LaunchBlock.x(32))

  def warpSumRounds = kernel("quotedWarpSumRounds", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"), value[Int]("rounds"))) { p =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    var total = 0.0
    for round <- deviceRange(0, p._4) do
      val item = if i < p._3 then p._1(i) else -0.5
      val sum = warpReduceSum(sync.fullMask, item, 8)
      total += sum
    if i < p._3 then p._2(i) = total
  }.requiringBlock(LaunchBlock.x(32))

  def blockSum = kernel("quotedBlockSum", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { (data, target, count) =>
    val scratch = blockReduction[Float](256)
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val item = if i < count then data(i) else 0.0f
    val total = scratch.sum(item)
    if i < count then target(i) = total
  }.requiringBlock(LaunchBlock.x(256))

  def blockTreeMax = kernel("quotedBlockMax", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"))) { p =>
    val scratch = blockReduction[Int](64)
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val item = if i < p._3 then p._1(i) else -1000000
    val best = scratch.reduceTree(item)((a, b) => if a > b then a else b)
    if i < p._3 then p._2(i) = best
  }.requiringBlock(LaunchBlock.x(64))

  def blockSumReuse = kernel("quotedBlockReuse", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"))) { p =>
    val scratch = blockReduction[Double](128)
    val alias = scratch
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val item = if i < p._3 then p._1(i) else 0.25
    val total = scratch.sum(item)
    val smallest = alias.reduceTree(item)((a, b) => if a < b then a else b)
    if i < p._3 then p._2(i) = total - smallest
  }.requiringBlock(LaunchBlock.x(128))

  def shuffleRotate = kernel("quotedShuffleRotate", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"))) { p =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    val item = if i < p._3 then p._1(i) else -7
    val first = warpShuffle(sync.fullMask, item, 0)
    val source = (lane + 1) % 32
    val next = warpShuffle(sync.fullMask, item, source)
    if i < p._3 then p._2(i) = first + next
  }.requiringBlock(LaunchBlock.x(32))

  def shufflePrefix = kernel("quotedShufflePrefix", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { (data, target, count) =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    var total = if i < count then data(i) else 0.0f
    val add1 = warpShuffleUp(sync.fullMask, total, 1)
    if lane >= 1 then total += add1
    val add2 = warpShuffleUp(sync.fullMask, total, 2)
    if lane >= 2 then total += add2
    val add4 = warpShuffleUp(sync.fullMask, total, 4)
    if lane >= 4 then total += add4
    val add8 = warpShuffleUp(sync.fullMask, total, 8)
    if lane >= 8 then total += add8
    val add16 = warpShuffleUp(sync.fullMask, total, 16)
    if lane >= 16 then total += add16
    if i < count then target(i) = total
  }.requiringBlock(LaunchBlock.x(32))

  def shuffleFold = kernel("quotedShuffleFold", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"))) { p =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    val item = if i < p._3 then p._1(i) else 0.5
    val down = warpShuffleDown(sync.fullMask, item, 2, 8)
    val partner = warpShuffleXor(sync.fullMask, item, 1, 8)
    if i < p._3 then p._2(i) = down - partner
  }.requiringBlock(LaunchBlock.x(32))

  def atomicHistogram = kernel("quotedAtomicHistogram", params(input[Int]("data"), inOut[Int]("bins"),
      value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._3 then
      val bin = ((p._1(i) % 8) + 8) % 8
      atomic(p._2(bin)).add(1)
  }

  def atomicCompact = kernel("quotedAtomicCompact", params(input[Float]("data"), output[Float]("packed"),
      inOut[Int]("cursor"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._4 && p._1(i) > 0.0f then
      val slot = atomic(p._3(0)).fetchAdd(1)
      p._2(slot) = p._1(i)
  }

  def atomicSharedSum = kernel("quotedAtomicSharedSum", params(input[Float]("data"), output[Float]("totals"),
      value[Int]("count"))) { (data, totals, count) =>
    val tile = sharedArray[Float](1)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    sync.blockAfter { if lane == 0 then tile(0) = 0.0f }
    val item = if i < count then data(i) else 0.0f
    atomic(tile(0)).add(item)
    sync.block()
    if lane == 0 then totals(blockIdx.x) = tile(0)
  }.requiringBlock(LaunchBlock.x(64))

  def scopedFlag = kernel("quotedScopedFlag", params(input[Int]("data"), inOut[Int]("cell"),
      inOut[Int]("flag"), output[Int]("out"), value[Int]("count"))) { p =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    if lane == 0 then
      p._2(blockIdx.x) = if i < p._5 then p._1(i) else -5
      atomicDevice(p._3(blockIdx.x)).store(1, MemoryOrder.Release)
    sync.block()
    val ready = atomicDevice(p._3(blockIdx.x)).load(MemoryOrder.Acquire)
    if i < p._5 then p._4(i) = if ready == 1 then p._2(blockIdx.x) else -9
  }

  def scopedExchange = kernel("quotedScopedExchange", params(input[Int]("data"), inOut[Int]("cells"),
      output[Int]("out"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._4 then
      val previous = atomicDevice(p._2(i)).exchange(p._1(i), MemoryOrder.Relaxed)
      val before = atomicDevice(p._2(i)).fetchSub(previous, MemoryOrder.Relaxed)
      p._3(i) = previous + before
  }

  def scopedSharedCounter = kernel("quotedScopedCounter", params(input[Double]("data"), output[Double]("totals"),
      value[Int]("count"))) { (data, totals, count) =>
    val tile = sharedArray[Double](1)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    sync.blockAfter { if lane == 0 then atomicBlock(tile(0)).store(0.0, MemoryOrder.Relaxed) }
    val item = if i < count then data(i) else 0.25
    val before = atomicBlock(tile(0)).fetchAdd(item, MemoryOrder.Relaxed)
    sync.block()
    val total = atomicBlock(tile(0)).load(MemoryOrder.Acquire)
    if lane == 0 then totals(blockIdx.x) = total + before * 0.0
  }.requiringBlock(LaunchBlock.x(64))

  def scopedBounds = kernel("quotedScopedBounds", params(input[Int]("data"), inOut[Int]("floors"),
      inOut[Int]("ceilings"), output[Int]("out"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._5 then
      val belowOld = atomicDevice(p._2(i)).fetchMin(p._1(i), MemoryOrder.Relaxed)
      val aboveOld = atomicDevice(p._3(i)).fetchMax(p._1(i), MemoryOrder.Relaxed)
      p._4(i) = belowOld + aboveOld
  }

  def scopedBitmask = kernel("quotedScopedBitmask", params(input[Int]("data"), inOut[Int]("masks"),
      output[Int]("andOld"), output[Int]("orOld"), output[Int]("xorOld"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._6 then
      val cleared = atomicDevice(p._2(i)).fetchAnd(p._1(i), MemoryOrder.Relaxed)
      val merged = atomicDevice(p._2(i)).fetchOr(5, MemoryOrder.Relaxed)
      val flipped = atomicDevice(p._2(i)).fetchXor(3, MemoryOrder.Relaxed)
      p._3(i) = cleared
      p._4(i) = merged
      p._5(i) = flipped
  }

  def scopedClaim = kernel("quotedScopedClaim", params(input[Int]("data"), inOut[Int]("cells"),
      inOut[Int]("winners"), output[Int]("out"), value[Int]("count"))) { p =>
    val gate = sharedArray[Int](1)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    sync.blockAfter { if lane == 0 then atomicBlock(gate(0)).store(0, MemoryOrder.Relaxed) }
    val ticket = atomicBlock(gate(0)).compareExchange(0, 1, MemoryOrder.AcquireRelease, MemoryOrder.Acquire)
    if ticket == 0 then
      val slot = atomicDevice(p._3(blockIdx.x)).fetchAdd(1, MemoryOrder.Relaxed)
      p._4(p._5 + 31) = slot
    if i < p._5 then
      val current = p._2(i)
      val expected = if i % 2 == 0 then current else current + 1
      val witnessed = atomicBlock(p._2(i)).compareExchange(expected, p._1(i), MemoryOrder.Relaxed, MemoryOrder.Relaxed)
      p._4(i) = witnessed
  }.requiringBlock(LaunchBlock.x(64))

  def mathActivation = kernel("quotedMathActivation", params(input[Float]("data"), output[Float]("target"),
      value[Float]("mean"), value[Float]("variance"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._5 then
      val scale = rsqrt(p._4 + 0.00001f)
      p._2(i) = tanh((p._1(i) - p._3) * scale)
  }

  def mathLogistic = kernel("quotedMathLogistic", params(input[Double]("data"), output[Double]("sigmoid"),
      output[Double]("logs"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._4 then
      p._2(i) = 1.0 / (1.0 + exp(0.0 - p._1(i)))
      p._3(i) = log(p._1(i) * p._1(i) + 1.0)
  }

  def mathDistance = kernel("quotedMathDistance", params(input[Float]("xs"), input[Float]("ys"),
      output[Float]("out"), value[Float]("cutoff"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._5 then
      val radius = sqrt(p._1(i) * p._1(i) + p._2(i) * p._2(i))
      p._3(i) = if radius > p._4 then radius else log(radius + 1.0f)
  }

  def bitPack = kernel("quotedBitPack", params(input[Int]("data"), output[Int]("packed"),
      value[Int]("rotate"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._4 then
      val rotated = (p._1(i) << p._3) | (p._1(i) >>> (32 - p._3))
      val arithmetic = p._1(i) >> 2
      p._2(i) = rotated ^ ~arithmetic
  }

  def negateBlend = kernel("quotedNegateBlend", params(input[Float]("floats"), output[Float]("outF"),
      input[Double]("doubles"), output[Double]("outD"), value[Float]("weight"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    if i < p._6 then
      p._2(i) = -p._1(i) * p._5 + -(p._1(i) - p._5)
      p._4(i) = -p._3(i) * 0.5 + -(p._3(i) * p._3(i))
  }

  def sharedExchange = kernel("quotedSharedExchange", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { (data, target, count) =>
    val tile = sharedArray[Float](64)
    val alias = tile
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    alias(lane) = if i < count then data(i) else 0.0f
    barrier()
    if i < count then target(i) = tile((lane + 1) % 64)
  }.requiringBlock(LaunchBlock.x(64))

  def sharedReuse = kernel("quotedSharedReuse", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("rounds"))) { p =>
    val tile = sharedArray[Int](64)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    tile(lane) = if i < p._3 then p._1(i) else 0
    barrier()
    for round <- deviceRange(0, p._4) do
      val alias = tile
      val previous = alias((lane + 1) % 64)
      barrier()
      alias(lane) = previous + round
      barrier()
    if i < p._3 then p._2(i) = tile(lane)
  }.requiringBlock(LaunchBlock.x(64))

  def sharedFlags = kernel("quotedSharedFlags", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"), value[Boolean]("enabled"))) { (data, target, count, enabled) =>
    val flags = sharedArray[Boolean](64)
    val values = sharedArray[Double](64)
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    flags(lane) = enabled && i < count
    values(lane) = if flags(lane) then data(i) else 0.0
    barrier()
    val neighbor = (lane + 1) % 64
    if i < count then target(i) = if flags(neighbor) then values(neighbor) else -7.0
  }.requiringBlock(LaunchBlock.x(64))

  def sharedTile = kernel("quotedSharedTile", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { p =>
    val tile = sharedArray2D[Float](2, 32)
    val lane = threadIdx.x
    val row = lane / 32
    val column = lane % 32
    val i = blockIdx.x * blockDim.x + lane
    tile(row, column) = if i < p._3 then p._1(i) else 0.0f
    barrier()
    if i < p._3 then p._2(i) = tile(row, (column + 1) % 32)
  }.requiringBlock(LaunchBlock.x(64))

  def sharedCube = kernel("quotedSharedCube", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("rounds"))) { p =>
    val cube = sharedArray3D[Int](2, 2, 16)
    val lane = threadIdx.x
    val layer = lane / 32
    val row = (lane / 16) % 2
    val column = lane % 16
    val i = blockIdx.x * blockDim.x + lane
    cube(layer, row, column) = if i < p._3 then p._1(i) else 0
    barrier()
    for round <- deviceRange(0, p._4) do
      val previous = cube((layer + 1) % 2, row, column)
      barrier()
      cube(layer, row, column) = previous + round
      barrier()
    if i < p._3 then p._2(i) = cube(layer, row, column)
  }.requiringBlock(LaunchBlock.x(64))

  def warpTreeMax = kernel("quotedWarpTreeMax", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { p =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    val item = if i < p._3 then p._1(i) else -1000.0f
    val best = warpReduceTree(sync.fullMask, item)((a, b) => if a > b then a else b)
    if i < p._3 then p._2(i) = best
  }.requiringBlock(LaunchBlock.x(32))

  def warpTreeDifference = kernel("quotedWarpTreeDifference", params(input[Int]("data"), output[Int]("target"),
      value[Int]("count"))) { p =>
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    val item = if i < p._3 then p._1(i) else 0
    val folded = warpReduceTree(sync.fullMask, item, 8)((a, b) => a - b)
    if i < p._3 then p._2(i) = folded
  }.requiringBlock(LaunchBlock.x(32))

  def dynamicExchange = kernel("quotedDynamicExchange", params(input[Float]("data"), output[Float]("target"),
      value[Int]("count"))) { p =>
    val tile = dynamicSharedArray[Float]()
    val lane = threadIdx.x
    val i = blockIdx.x * blockDim.x + lane
    tile(lane) = if i < p._3 then p._1(i) else 0.0f
    barrier()
    if i < p._3 then p._2(i) = tile((lane + 1) % 64)
  }.requiringBlock(LaunchBlock.x(64))

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

  def tupleRows = kernel("quotedTupleRows", params(input[Float]("data"), output[Float]("target"),
      value[Int]("rows"), value[Int]("columns"), value[Float]("threshold"), value[Float]("initial"))) { p =>
    val row = blockIdx.x * blockDim.x + threadIdx.x
    if row < p._3 then
      val pairs = for column <- deviceRange(-1, p._4 + 1) if column >= 0 if column < p._4
        yield (column, p._1(row * p._4 + column), column % 2 == 0)
      val total = pairs.withFilter(pair => pair._3 && pair._2 > p._5)
        .map(pair => (pair._2 * 2.0f, pair._1))
        .foldLeft(p._6)((sum, pair) => sum - pair._1 - (if pair._2 == 0 then 0.5f else 0.0f))
      p._2(row) = total
  }

  def tupleReuse = kernel("quotedTupleReuse", params(inOut[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var begin = p._4
      var end = p._5
      var total = 0
      val pairs = deviceRange(begin, end).map(i => (p._1(lane) + i, p._1(lane), i))
        .withFilter(pair => pair._1 >= 0)
      val alias = pairs
      begin = 0
      end = 0
      alias.foreach { pair =>
        val saved = pair
        p._1(lane) = saved._1 + 1
        total += saved._2 + saved._2 + saved._3
      }
      val result = pairs.foldLeft(total)((sum, pair) => sum - pair._1 - pair._2)
      p._2(lane) = result
  }

  def tupleNested = kernel("quotedTupleNested", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      val fixed = deviceRange(0, 2)
      val pairs = deviceRange(0, p._4).map(i => (i, p._1(lane), p._5))
      val values = pairs.flatMap(pair => deviceRange(0, pair._1).withFilter(i => i % 2 == 0)
        .map(i => (pair._2 / 2.0, pair._3, i + pair._1)))
        .withFilter(pair => pair._2 && pair._3 > 0)
        .map(pair => pair)
        .flatMap(pair => fixed.map(k => if k == 0 then pair._1 else pair._1 + 1.0))
      val result = values.foldLeft(p._1(lane))((sum, item) => sum / 2.0 - item)
      p._2(lane) = result
  }

  def tupleFoldRows = kernel("quotedTupleFoldRows", params(input[Float]("data"), output[Float]("target"),
      output[Int]("visits"), value[Int]("rows"), value[Int]("columns"), value[Float]("seed"))) { p =>
    val row = blockIdx.x * blockDim.x + threadIdx.x
    if row < p._4 then
      val pairs = for column <- deviceRange(-1, p._5 + 1) if column >= 0 if column < p._5
        yield (column, p._1(row * p._5 + column))
      val result = pairs.withFilter(pair => pair._1 % 2 == 0 && pair._2 != 0.0f)
        .foldLeft((p._6, 0))((state, pair) =>
          (state._1 - pair._2 - (if state._2 % 2 == 0 then 0.5f else 0.0f), state._2 + 1))
      p._2(row) = result._1
      p._3(row) = result._2
  }

  def tupleFoldReuse = kernel("quotedTupleFoldReuse", params(inOut[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var begin = p._4
      var end = p._5
      val values = deviceRange(begin, end).withFilter(i => i % 2 == 0).map(i => p._1(lane) + i)
      begin = 0
      end = 0
      val first = values.foldLeft((p._1(lane), p._1(lane) + 1))((state, item) => (state._2, state._1 - item))
      val saved = first
      p._1(lane) = saved._1
      val second = values.foldLeft(saved)((state, item) => (state._2 + item, state._1))
      p._1(lane) = first._1 + second._2
      p._2(lane) = saved._2 + second._1
  }

  def tupleFoldNested = kernel("quotedTupleFoldNested", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var total = p._1(lane)
      for round <- 0 until p._4 do
        val pairs = deviceRange(0, round).map(i => (i, p._1(lane), p._5))
          .flatMap(pair => deviceRange(0, pair._1).map(j => (j, pair._2, pair._3)))
          .withFilter(pair => pair._3)
        val result = pairs.foldLeft((total, false))((state, pair) =>
          (if state._2 then state._1 / 2.0 - pair._2 else state._1 - pair._2, (state._1 > 0.0) != state._2))
        total = result._1 + (if result._2 then 0.25 else 0.5)
      p._2(lane) = total
  }

  def namedFoldRows = kernel("quotedNamedFoldRows", params(input[Float]("data"), output[Float]("target"),
      output[Int]("visits"), value[Int]("rows"), value[Int]("columns"), value[Float]("seed"))) { p =>
    val row = blockIdx.x * blockDim.x + threadIdx.x
    if row < p._4 then
      val items = for column <- deviceRange(-1, p._5 + 1) if column >= 0 if column < p._5
        yield (index = column, value = p._1(row * p._5 + column))
      val result = items.withFilter(item => item.index % 2 == 0 && item.value != 0.0f)
        .foldLeft((sum = p._6, count = 0))((state, item) =>
          (sum = state.sum - item.value - (if state.count % 2 == 0 then 0.5f else 0.0f), count = state.count + 1))
      p._2(row) = result.sum
      p._3(row) = result.count
  }

  def namedFoldReuse = kernel("quotedNamedFoldReuse", params(inOut[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var begin = p._4
      var end = p._5
      val values = deviceRange(begin, end).withFilter(i => i % 2 == 0).map(i => p._1(lane) + i)
      begin = 0
      end = 0
      val first = values.foldLeft((left = p._1(lane), right = p._1(lane) + 1))((state, item) => (left = state.right, right = state.left - item))
      val saved = first
      p._1(lane) = saved.left
      val second = values.foldLeft(saved)((state, item) => (left = state.right + item, right = state.left))
      p._1(lane) = first.left + second.right
      p._2(lane) = saved.right + second.left
  }

  def namedFoldNested = kernel("quotedNamedFoldNested", params(input[Double]("data"), output[Double]("target"),
      value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var total = p._1(lane)
      for round <- 0 until p._4 do
        val items = deviceRange(0, round).map(i => (index = i, value = p._1(lane), enabled = p._5))
          .flatMap(item => deviceRange(0, item.index).map(j => (index = j, value = item.value, enabled = item.enabled)))
          .withFilter(item => item.enabled)
        val result = items.foldLeft((sum = total, flag = false))((state, item) =>
          (sum = if state.flag then state.sum / 2.0 - item.value else state.sum - item.value, flag = (state.sum > 0.0) != state.flag))
        total = result.sum + (if result.flag then 0.25 else 0.5)
      p._2(lane) = total
  }

  def namedReuse = kernel("quotedNamedReuse", params(inOut[Int]("data"), output[Int]("target"),
      value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
    val lane = blockIdx.x * blockDim.x + threadIdx.x
    if lane < p._3 then
      var begin = p._4
      var end = p._5
      var total = 0
      val items = deviceRange(begin, end).map(i => (next = p._1(lane) + i, before = p._1(lane), index = i))
        .withFilter(item => item.next >= 0)
      val alias = items
      begin = 0
      end = 0
      alias.foreach { item =>
        val saved = item
        p._1(lane) = saved.next + 1
        total += saved.before + saved.before + saved.index
      }
      val result = items.foldLeft(total)((sum, item) => sum - item.next - item.before)
      p._2(lane) = result
  }
