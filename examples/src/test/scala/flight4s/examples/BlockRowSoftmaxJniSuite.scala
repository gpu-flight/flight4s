package flight4s.examples

import munit.FunSuite

class BlockRowSoftmaxJniSuite extends FunSuite:
  test("block softmax matches a Double reference across sub-warp and multi-stride rows"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val sizes = Vector(1, 2, 31, 32, 33, 127, 128, 129, 257, 1025, 4097)
    val regular = sizes.map { columns =>
      val rows = 7
      val input = Array.tabulate(rows * columns) { i =>
        val row = i / columns
        if row == 0 then 0.0f
        else if row == 1 then 1000.0f + (i % columns) / 16.0f
        else if row == 2 then -1000.0f + (i % columns) / 16.0f
        else (i % 101 - 50) / 8.0f
      }
      (rows, columns, input)
    }
    val fixtures = regular ++ Vector(
      (1, 4, Array(Float.MaxValue, -Float.MaxValue, 0.0f, Float.MaxValue)),
      (129, 257, Array.tabulate(129 * 257)(i => (i % 53 - 26) / 4.0f))
    )
    fixtures.foreach { (rows, columns, input) =>
      val unchanged = input.clone()
      val actual = BlockRowSoftmax.run(input, rows, columns)
      val expected = input.grouped(columns).flatMap { row =>
        val maximum = row.map(_.toDouble).max
        val weights = row.map(x => math.exp(x.toDouble - maximum))
        val denominator = weights.sum
        weights.map(_ / denominator)
      }.toVector
      assertEquals(input.toVector, unchanged.toVector)
      assertEquals(actual.length, input.length)
      actual.zip(expected).zipWithIndex.foreach { case ((value, reference), index) =>
        assert(java.lang.Float.isFinite(value) && value >= 0.0f, s"$rows x $columns index $index: $value")
        assert(math.abs(value.toDouble - reference) <= 2e-6, s"$rows x $columns index $index: $value != $reference")
      }
      actual.grouped(columns).foreach(row => assert(math.abs(row.map(_.toDouble).sum - 1.0) <= 2e-5))
      val repeated = BlockRowSoftmax.run(input, rows, columns)
      assertEquals(repeated.toVector, actual.toVector)
    }
