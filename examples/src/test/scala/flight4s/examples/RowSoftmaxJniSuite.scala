package flight4s.examples

import munit.FunSuite

class RowSoftmaxJniSuite extends FunSuite:
  test("Scala softmax matches a Double CPU reference on CUDA"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val fixtures = Vector(
      (1, 1, Array(42.0f)),
      (3, 4, Array.fill(12)(0.0f)),
      (2, 4, Array(1000.0f, 1001.0f, 1002.0f, 1003.0f, -1000.0f, -999.0f, -998.0f, -997.0f)),
      (1, 4, Array(Float.MaxValue, -Float.MaxValue, 0.0f, Float.MaxValue)),
      (129, 7, Array.tabulate(129 * 7)(i => (i % 29 - 14) / 8.0f)),
      (257, 33, Array.tabulate(257 * 33)(i => (i % 101 - 50) / 8.0f))
    )
    fixtures.foreach { (rows, columns, input) =>
      val actual = RowSoftmax.run(input, rows, columns)
      val expected = input.grouped(columns).flatMap { row =>
        val maximum = row.map(_.toDouble).max
        val weights = row.map(value => math.exp(value.toDouble - maximum))
        val denominator = weights.sum
        weights.map(_ / denominator)
      }.toVector
      assertEquals(actual.length, input.length)
      actual.zip(expected).zipWithIndex.foreach { case ((value, reference), index) =>
        assert(java.lang.Float.isFinite(value) && value >= 0.0f,
          s"$rows x $columns index $index: invalid probability $value")
        assert(math.abs(value.toDouble - reference) <= 2e-6,
          s"$rows x $columns index $index: obtained $value, expected $reference")
      }
      actual.grouped(columns).foreach { row =>
        assert(math.abs(row.map(_.toDouble).sum - 1.0) <= 2e-5)
      }
    }
