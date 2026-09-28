package flight4s.examples

import munit.FunSuite

class BlockRowStatisticsJniSuite extends FunSuite:
  test("cooperative moments match decimal references across empty lanes and uneven partial states"):
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val sizes = Vector(1, 2, 31, 32, 33, 127, 128, 129, 257, 1025, 4097, 100001)
    val fixtures = sizes.map { columns =>
      val rows = 6
      val values = Array.tabulate(rows * columns) { i =>
        val column = i % columns
        (i / columns) match
          case 0 => -7.0f
          case 1 => 1000000.0f + (column % 101 - 50) / 8.0f
          case 2 => if column % 2 == 0 then Float.MaxValue else -Float.MaxValue
          case 3 => (column % 5 - 2) * java.lang.Float.MIN_VALUE
          case 4 => (column % 29 - 14) / 8.0f
          case _ => -1000000.0f + ((columns - 1 - column) % 101 - 50) / 8.0f
      }
      (rows, columns, values)
    } :+ (129, 257, Array.tabulate(129 * 257)(i => (i % 53 - 26) / 4.0f))
    var checked = 0
    fixtures.foreach { (rows, columns, values) =>
      val original = values.map(java.lang.Float.floatToRawIntBits).toVector
      val expected = values.grouped(columns).map(RowStatisticsReference.apply).toVector
      val actual = BlockRowStatistics.run(values, rows, columns)
      assertEquals(actual.means.length, rows)
      assertEquals(actual.populationVariances.length, rows)
      values.grouped(columns).zipWithIndex.foreach { (row, index) =>
        val (mean, variance) = expected(index)
        val maxAbs = row.iterator.map(x => math.abs(x.toDouble)).max
        val meanTolerance = math.max(maxAbs * 1e-12, java.lang.Double.MIN_VALUE)
        val varianceTolerance = math.max(math.abs(variance) * 1e-9, java.lang.Double.MIN_VALUE)
        assert(java.lang.Double.isFinite(actual.means(index)))
        assert(java.lang.Double.isFinite(actual.populationVariances(index)) && actual.populationVariances(index) >= 0.0)
        assert(math.abs(actual.means(index) - mean) <= meanTolerance,
          s"$rows/$columns/$index mean ${actual.means(index)} != $mean")
        assert(math.abs(actual.populationVariances(index) - variance) <= varianceTolerance,
          s"$rows/$columns/$index variance ${actual.populationVariances(index)} != $variance")
        checked += 2
      }
      val repeated = BlockRowStatistics.run(values, rows, columns)
      assertEquals(repeated.means.toVector, actual.means.toVector)
      assertEquals(repeated.populationVariances.toVector, actual.populationVariances.toVector)
      assertEquals(values.map(java.lang.Float.floatToRawIntBits).toVector, original)
    }
    assertEquals(checked, 402)
