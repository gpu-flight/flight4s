package flight4s.examples

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class BlockRowLayerNormSuite extends FunSuite:
  test("block layer normalization exposes a separate typed entry point"):
    assertEquals(typeCheckErrors("""
      import flight4s.examples.BlockRowLayerNorm
      val out: Array[Float] = BlockRowLayerNorm.run(Array(1.0f, 3.0f),
        Array(2.0f, -4.0f), Array(0.25f, 1.0f), 1, 2, epsilon = 3.0)
      BlockRowLayerNorm.main(Array("--cuda-source"))
    """), Nil)

  test("host validation precedes CUDA and empty batches stay native-free"):
    assertEquals(BlockRowLayerNorm.run(Array.emptyFloatArray, Array(1.0f), Array(0.0f),
      0, 1, deviceOrdinal = -1).toVector, Vector.empty)
    for (values, rows, columns) <- Vector((Array.emptyFloatArray, -1, 1),
      (Array.emptyFloatArray, 1, 0), (Array(1.0f), 1, 2), (Array.emptyFloatArray, 50000, 50000)) do
      intercept[IllegalArgumentException](BlockRowLayerNorm.run(values, Array(1.0f), Array(0.0f),
        rows, columns, deviceOrdinal = -1))
    for invalid <- Vector(Float.NaN, Float.PositiveInfinity, Float.NegativeInfinity) do
      for (values, gain, bias) <- Vector((Array(invalid), Array(1.0f), Array(0.0f)),
        (Array(1.0f), Array(invalid), Array(0.0f)), (Array(1.0f), Array(1.0f), Array(invalid))) do
        intercept[IllegalArgumentException](BlockRowLayerNorm.run(values, gain, bias, 1, 1, deviceOrdinal = -1))
    for epsilon <- Vector(0.0, -1.0, Double.NaN, Double.PositiveInfinity) do
      intercept[IllegalArgumentException](BlockRowLayerNorm.run(Array(1.0f), Array(1.0f), Array(0.0f),
        1, 1, epsilon, deviceOrdinal = -1))
    intercept[IllegalArgumentException](BlockRowLayerNorm.run(Array.emptyFloatArray,
      Array.emptyFloatArray, Array(0.0f), 0, 1, deviceOrdinal = -1))

  test("one exact 128-thread block owns each row"):
    assertEquals(BlockRowLayerNorm.definition.requiredBlock, Some(LaunchBlock.x(128)))
    assertEquals(BlockRowLayerNorm.generated.launchRequirements.requiredBlock, Some(LaunchBlock.x(128)))
    for rows <- Vector(1, 129, Int.MaxValue) do
      assertEquals(BlockRowLayerNorm.launchConfig(rows), LaunchConfig(Grid.x(rows), LaunchBlock.x(128)))

  test("output follows eight uniform barriers and uses strided columns"):
    val validation = KernelValidator.validate(BlockRowLayerNorm.definition)
    assertEquals(validation.errors, Vector.empty)
    assertEquals(validation.warnings, Vector.empty)
    val row = BlockRowLayerNorm.definition.body.statements.last.asInstanceOf[IfThen].thenBlock
    assertEquals(row.statements.count(_.isInstanceOf[Barrier]), 8)
    assert(row.statements.last.isInstanceOf[ForLoop])
    val output = row.statements.last.asInstanceOf[ForLoop]
    assertEquals(output.body.statements.size, 1)
    assert(output.body.statements.head.isInstanceOf[Store[?, ?]])
    val source = BlockRowLayerNorm.generated.cudaSource
    assert(source.contains("__shared__ int counts[128];"))
    assert(source.contains("__shared__ double partialMeans[128];"))
    assert(source.contains("__shared__ double partialM2[128];"))
    assertEquals(source.sliding("+= 128LL".length).count(_ == "+= 128LL"), 2)
    assert(source.lastIndexOf("__syncthreads();") < source.indexOf("double rowMean"))
    assert(source.contains("::__double2float_rn("))

  test("the statistics merge is shared without changing its staged structure"):
    val stats = BlockRowStatistics.definition.body.statements.last.asInstanceOf[IfThen].thenBlock
    val norm = BlockRowLayerNorm.definition.body.statements.last.asInstanceOf[IfThen].thenBlock
    // Source locations belong to each caller; statement kinds and counts stay identical.
    assertEquals(norm.statements.take(stats.statements.size - 1).map(_.getClass),
      stats.statements.dropRight(1).map(_.getClass))
    val statsSource = BlockRowStatistics.generated.cudaSource
    val normSource = BlockRowLayerNorm.generated.cudaSource
    val start = "__syncthreads();"
    val end = "__syncthreads();"
    def merge(source: String): String =
      source.substring(source.indexOf(start), source.lastIndexOf(end) + end.length)
    assertEquals(merge(normSource), merge(statsSource))

  test("source inspection and CLI errors need no CUDA"):
    val output = new java.io.ByteArrayOutputStream
    Console.withOut(output) { BlockRowLayerNorm.main(Array("--cuda-source")) }
    assert(output.toString("UTF-8").contains(BlockRowLayerNorm.generated.cudaSource))
    intercept[IllegalArgumentException](BlockRowLayerNorm.main(Array("--invalid")))
