package flight4s.examples

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class BlockRowStatisticsSuite extends FunSuite:
  test("block-cooperative statistics has a separate runnable entry point"):
    assertEquals(typeCheckErrors("""
      import flight4s.examples.{BlockRowStatistics, RowStatistics}
      val result: RowStatistics.Result = BlockRowStatistics.run(Array.emptyFloatArray, 0, 3)
      BlockRowStatistics.main(Array("--cuda-source"))
    """), Nil)

  test("host contracts reject invalid input and empty batches do not open CUDA"):
    for (values, rows, columns) <- Vector((Array.emptyFloatArray, -1, 3),
      (Array.emptyFloatArray, 1, 0), (Array(1.0f), 1, 2), (Array.emptyFloatArray, 50000, 50000)) do
      intercept[IllegalArgumentException](BlockRowStatistics.run(values, rows, columns, deviceOrdinal = -1))
    for invalid <- Vector(Float.NaN, Float.PositiveInfinity, Float.NegativeInfinity) do
      intercept[IllegalArgumentException](BlockRowStatistics.run(Array(invalid), 1, 1, deviceOrdinal = -1))
    val empty = BlockRowStatistics.run(Array.emptyFloatArray, 0, 3, deviceOrdinal = -1)
    assertEquals(empty.means.toVector, Vector.empty)
    assertEquals(empty.populationVariances.toVector, Vector.empty)

  test("launch shape is exactly one 128-thread block per row"):
    assertEquals(BlockRowStatistics.definition.requiredBlock, Some(LaunchBlock.x(128)))
    assertEquals(BlockRowStatistics.generated.launchRequirements.requiredBlock, Some(LaunchBlock.x(128)))
    for rows <- Vector(1, 3, 129, Int.MaxValue) do
      assertEquals(BlockRowStatistics.launchConfig(rows), LaunchConfig(Grid.x(rows), LaunchBlock.x(128)))

  test("three typed shared arrays and eight uniform barriers define the merge tree"):
    val validation = KernelValidator.validate(BlockRowStatistics.definition)
    assertEquals(validation.errors, Vector.empty)
    assertEquals(validation.warnings, Vector.empty)
    val source = BlockRowStatistics.generated.cudaSource
    assert(source.contains("__shared__ int counts[128];"))
    assert(source.contains("__shared__ double partialMeans[128];"))
    assert(source.contains("__shared__ double partialM2[128];"))
    assert(source.contains("+= 128LL"))
    assertEquals(source.sliding("__syncthreads();".length).count(_ == "__syncthreads();"), 8)
    assertEquals(BlockRowStatistics.generated.compilerOptions.additionalNvrtcOptions, Vector.empty)

  test("merge snapshots precede all shared stores and empty states are guarded"):
    val row = BlockRowStatistics.definition.body.statements.collectFirst { case b: IfThen => b.thenBlock }.get
    val phases = row.statements.collect { case b: IfThen => b }.dropRight(1)
    assertEquals(phases.size, 7)
    assertEquals(row.statements.count(_.isInstanceOf[Barrier]), 8)
    phases.foreach { phase =>
      val guarded = phase.thenBlock.statements.last.asInstanceOf[IfThen]
      val body = guarded.thenBlock.statements
      assertEquals(body.size, 11)
      assert(body.take(8).forall(_.isInstanceOf[LocalDeclaration[?]]))
      assert(body.drop(8).forall(_.isInstanceOf[Store[?, ?]]))
      assert(!phase.thenBlock.statements.exists(_.isInstanceOf[Barrier]))
      assert(!body.exists(_.isInstanceOf[Barrier]))
    }

  test("source inspection and usage errors are available without CUDA"):
    val output = new java.io.ByteArrayOutputStream
    Console.withOut(output) { BlockRowStatistics.main(Array("--cuda-source")) }
    assert(output.toString("UTF-8").contains(BlockRowStatistics.generated.cudaSource))
    intercept[IllegalArgumentException](BlockRowStatistics.main(Array("--invalid")))
