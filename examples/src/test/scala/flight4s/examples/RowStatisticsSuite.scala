package flight4s.examples

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.ir.*

class RowStatisticsSuite extends FunSuite:
  test("the example exposes Float input and separately typed Double statistics"):
    assertEquals(typeCheckErrors("""
      import flight4s.examples.RowStatistics
      val result: RowStatistics.Result = RowStatistics.run(Array(1.0f), rows = 1, columns = 1)
      val means: Array[Double] = result.means
      val variances: Array[Double] = result.populationVariances
    """), Nil)

  test("invalid shapes and non-finite inputs fail before native setup"):
    for (values, rows, columns) <- Vector(
      (Array.emptyFloatArray, -1, 3), (Array.emptyFloatArray, 1, 0),
      (Array.emptyFloatArray, 0, -1), (Array(1.0f), 1, 2),
      (Array.emptyFloatArray, 50000, 50000), (Array(1.0f), 0, 1)
    ) do intercept[IllegalArgumentException](RowStatistics.run(values, rows, columns, deviceOrdinal = -1))
    for invalid <- Vector(Float.NaN, Float.PositiveInfinity, Float.NegativeInfinity) do
      intercept[IllegalArgumentException](RowStatistics.run(Array(invalid), 1, 1, deviceOrdinal = -1))

  test("empty batches avoid CUDA and all finite Float magnitudes are accepted"):
    val result = RowStatistics.run(Array.emptyFloatArray, 0, Int.MaxValue, deviceOrdinal = -1)
    assertEquals(result.means.toVector, Vector.empty)
    assertEquals(result.populationVariances.toVector, Vector.empty)
    RowStatistics.validateInput(Array(Float.MaxValue, -Float.MaxValue, java.lang.Float.MIN_VALUE), 1, 3)

  test("launch coverage rounds up without overflowing"):
    for (rows, blocks) <- Vector(0 -> 0, 1 -> 1, 127 -> 1, 128 -> 1, 129 -> 2, Int.MaxValue -> 16777216) do
      assertEquals(RowStatistics.blockCount(rows), blocks)

  test("the kernel is one guarded serial tuple fold with no block collectives"):
    assertEquals(KernelValidator.validate(RowStatistics.definition).errors, Vector.empty)
    val guard = RowStatistics.definition.body.statements.last.asInstanceOf[IfThen]
    val loop = guard.thenBlock.statements.collect { case loop: ForLoop => loop }.head
    assertEquals(loop.body.statements.size, 6)
    assert(loop.body.statements.take(3).forall(_.isInstanceOf[LocalDeclaration[?]]))
    assert(loop.body.statements.drop(3).forall(_.isInstanceOf[Store[?, ?]]))
    val source = RowStatistics.generated.cudaSource
    assert(source.contains("static_cast<double>(values["))
    assert(source.contains("double moments_1"))
    assert(source.contains("double moments_2"))
    assert(source.contains("::__int2double_rn(moments_0)"))
    assert(!source.contains("__syncthreads"))
    assertEquals(RowStatistics.generated.compilerOptions.additionalNvrtcOptions, Vector.empty)

  test("source inspection and CLI argument errors require no GPU"):
    val output = new java.io.ByteArrayOutputStream
    Console.withOut(output) { RowStatistics.main(Array("--cuda-source")) }
    assert(output.toString("UTF-8").contains(RowStatistics.generated.cudaSource))
    intercept[IllegalArgumentException](RowStatistics.main(Array("--invalid")))
