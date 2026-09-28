package flight4s.examples

import java.math.{BigDecimal as Decimal, MathContext, RoundingMode}

private[examples] object RowStatisticsReference:
  private val precision = new MathContext(80, RoundingMode.HALF_EVEN)

  def apply(row: Array[Float]): (Double, Double) =
    val exact = row.map(x => new Decimal(x.toDouble))
    val count = Decimal.valueOf(row.length.toLong)
    val mean = exact.foldLeft(Decimal.ZERO)((sum, x) => sum.add(x)).divide(count, precision)
    val (residual, squaredDeviations) = exact.foldLeft((Decimal.ZERO, Decimal.ZERO)) { (sums, x) =>
      val delta = x.subtract(mean)
      (sums._1.add(delta), sums._2.add(delta.multiply(delta)))
    }
    // Correct for the rounded centering mean before the final division.
    val numerator = squaredDeviations.multiply(count).subtract(residual.multiply(residual))
    (mean.doubleValue(), numerator.divide(count.multiply(count), precision).doubleValue())
