package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaAutomaticBindingJniSuite extends FunSuite:
  test("automatic snapshots arrays loops and reusable block scratch execute on both stream paths"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(failure => fail(failure.message), identity)
    try
      for blockSize <- Vector(32, 128) do check(context, blockSize)
    finally context.close()

  private def check(context: CudaContext, blockSize: Int): Unit =
    val count = blockSize * 3
    val limit = count - 5
    val inputs = Array.tabulate(count)(i => i % 17 - 8)
    val definition = kernel("automatic", params(inOut[Int]("flight4s_auto_value_0"), output[Int]("out"), value[Int]("limit"))) { p =>
      val index = let { blockIdx.x * blockDim.x + threadIdx.x }
      val rank = let(threadIdx.x)
      val scratch = block.reduction[Int](LaunchBlock.x(blockSize))
      val tile = sharedArray2D[Int](2, blockSize, blockSize + 1)
      val cube = sharedArray3D[Int](2, 2, blockSize, blockSize + 1)
      val dynamic = dynamicSharedArray[Int]()
      val temporary = localArray[Int](2)
      val before = let(choose(index < p._3)(p._1(index).read)(literal(0)))
      when(index < p._3) { p._1(index) := literal(99) }
      tile(literal(0), rank) := before
      cube(literal(0), literal(0), rank) := tile(literal(0), rank).read
      dynamic(rank) := cube(literal(0), literal(0), rank).read
      temporary(literal(0)) := dynamic(rank).read
      val sum = scratch.sum(temporary(literal(0)).read)
      val difference = scratch.reduceTree(temporary(literal(0)).read)(_ - _)
      val loopTotal = local(literal(0))
      gpuFor(literal(0), literal(4), 2) { i => accumulate(loopTotal, i) }
      val rangeTotal = gpuRange(literal(0), literal(4)).map(identity).sum(literal(0))
      p._2(index * literal(4)) := before + before
      p._2(index * literal(4) + literal(1)) := sum
      p._2(index * literal(4) + literal(2)) := difference
      p._2(index * literal(4) + literal(3)) := loopTotal.read + rangeTotal
    }
    val generated = CudaCodegen.generate(definition).fold(error => fail(error.message), identity)
    val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated)), context.computeCapability, "automatic_bindings.cu")
      .fold(failure => fail(failure.message + "\n" + failure.compileLog), identity)
    val module = context.load(artifact).toOption.get
    val stream = context.createStream().toOption.get
    val source = context.allocate[Int](count).toOption.get
    val out = context.allocate[Int](count * 4 + 16).toOption.get
    try
      val function = module.function(generated).toOption.get
      val expected = Vector.tabulate(count * 4 + 16) { offset =>
        val index = offset / 4
        if index >= count then -999
        else
          val start = index / blockSize * blockSize
          val values = Vector.tabulate(blockSize)(i => if start + i < limit then inputs(start + i) else 0)
          offset % 4 match
            case 0 => if index < limit then inputs(index) * 2 else 0
            case 1 => values.sum
            case 2 => differenceTree(values)
            case _ => 8
      }
      for explicit <- Vector(false, true) do
        assertEquals(source.copyFrom(inputs), Right(()))
        assertEquals(out.copyFrom(Array.fill(count * 4 + 16)(-999)), Right(()))
        val config = LaunchConfig(Grid.x(3), LaunchBlock.x(blockSize), dynamicSharedMemoryBytes = blockSize * 4)
        val args = definition.bind((source, out, limit))
        assertEquals(if explicit then function.launch(args, config, stream) else function.launch(args, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        assertEquals(out.copyToArray().toOption.get.toVector, expected, s"blockSize=$blockSize stream=$explicit")
        assertEquals(source.copyToArray().toOption.get.toVector,
          inputs.indices.map(i => if i < limit then 99 else inputs(i)).toVector)
    finally
      out.close()
      source.close()
      stream.close()
      module.close()

  private def differenceTree(values: Vector[Int]): Int =
    if values.size == 1 then values.head
    else differenceTree(values.grouped(2).map(pair => pair(0) - pair(1)).toVector)
