package flight4s.examples

import munit.FunSuite

class RowStatisticsJniSuite extends FunSuite:
  test("Welford tuple state matches independent high-precision population statistics on CUDA"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val tiny = java.lang.Float.MIN_VALUE
    val fixtures = Vector(
      (1, 1, Array(42.0f)),
      (4, 1, Array(0.0f, -0.0f, Float.MaxValue, -Float.MaxValue)),
      (3, 4, Array.fill(12)(-7.0f)),
      (2, 4, Array(1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f)),
      (2, 4, Array(1000000.0f, 1000001.0f, 1000002.0f, 1000003.0f,
        -1000003.0f, -1000002.0f, -1000001.0f, -1000000.0f)),
      (2, 4, Array(Float.MaxValue, -Float.MaxValue, 0.0f, Float.MaxValue,
        -Float.MaxValue, Float.MaxValue, Float.MaxValue, -Float.MaxValue)),
      (2, 4, Array(tiny, 2 * tiny, 3 * tiny, 4 * tiny, -tiny, -2 * tiny, -3 * tiny, -4 * tiny)),
      (129, 7, Array.tabulate(129 * 7)(i => (i % 29 - 14) / 8.0f)),
      (257, 33, Array.tabulate(257 * 33)(i => (i % 101 - 50) / 8.0f)),
      (3, 4097, Array.tabulate(3 * 4097)(i => 1000000.0f + (i % 101 - 50) / 8.0f)),
      (1, 257, Array.tabulate(257)(i => if i % 2 == 0 then Float.MaxValue else -Float.MaxValue))
    )
    var checked = 0
    fixtures.foreach { (rows, columns, values) =>
      val original = values.map(java.lang.Float.floatToRawIntBits).toVector
      val actual = RowStatistics.run(values, rows, columns)
      assertEquals(actual.means.length, rows)
      assertEquals(actual.populationVariances.length, rows)
      values.grouped(columns).zipWithIndex.foreach { (row, index) =>
        val (mean, variance) = RowStatisticsReference(row)
        val scale = row.iterator.map(x => math.abs(x.toDouble)).max
        val actualMean = actual.means(index)
        val actualVariance = actual.populationVariances(index)
        assert(java.lang.Double.isFinite(actualMean), s"non-finite mean at $rows/$columns/$index")
        assert(java.lang.Double.isFinite(actualVariance) && actualVariance >= 0.0,
          s"invalid variance at $rows/$columns/$index: $actualVariance")
        val meanTolerance = math.max(scale * 1e-12, java.lang.Double.MIN_VALUE)
        val varianceTolerance = math.max(math.abs(variance) * 1e-9, java.lang.Double.MIN_VALUE)
        assert(math.abs(actualMean - mean) <= meanTolerance,
          s"$rows/$columns/$index mean=$actualMean reference=$mean tolerance=$meanTolerance")
        assert(math.abs(actualVariance - variance) <= varianceTolerance,
          s"$rows/$columns/$index variance=$actualVariance reference=$variance tolerance=$varianceTolerance")
        checked += 2
      }
      assertEquals(values.map(java.lang.Float.floatToRawIntBits).toVector, original)
    }
    assertEquals(checked, 812)

  test("population variance uses N rather than N minus one"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val actual = RowStatistics.run(Array(1.0f, 2.0f, 3.0f, 4.0f), 1, 4)
    assertEquals(actual.means.toVector, Vector(2.5))
    assertEquals(actual.populationVariances.toVector, Vector(1.25))
