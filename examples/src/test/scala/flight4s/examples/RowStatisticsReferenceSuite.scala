package flight4s.examples

import munit.FunSuite

class RowStatisticsReferenceSuite extends FunSuite:
  test("decimal reference has exactly zero variance for subnormal singleton and constant rows"):
    for value <- Vector(java.lang.Float.MIN_VALUE, -2 * java.lang.Float.MIN_VALUE, Float.MaxValue, -7.0f)
        count <- Vector(1, 3, 17) do
      val (mean, variance) = RowStatisticsReference(Array.fill(count)(value))
      assertEquals(mean, value.toDouble)
      assertEquals(variance, 0.0, s"$value repeated $count times")

  test("decimal reference matches known population variance independently of input offset"):
    for offset <- Vector(0.0f, 1000000.0f, -1000000.0f) do
      val (mean, variance) = RowStatisticsReference(Array(1.0f, 2.0f, 3.0f, 4.0f).map(_ + offset))
      assertEquals(mean, offset.toDouble + 2.5)
      assertEquals(variance, 1.25)
