package flight4s.ml

/** Row-major dense Double matrix: values(row * columns + column). The caller keeps array ownership. */
final case class DenseMatrix(rows: Int, columns: Int, values: Array[Double])

private[ml] object MlValidation:
  def requireThat(condition: Boolean, message: => String): Unit =
    if !condition then throw IllegalArgumentException(message)

  /** Multiplies dimensions without overflow before any allocation decision. */
  def checkedSize(rows: Int, columns: Int, what: String): Int =
    val size = rows.toLong * columns.toLong
    requireThat(size <= Int.MaxValue, s"$what size $rows x $columns exceeds the supported array size")
    size.toInt

  def validateMatrix(x: DenseMatrix, what: String): Unit =
    requireThat(x.rows > 0, s"$what must have a positive row count, found ${x.rows}")
    requireThat(x.columns > 0, s"$what must have a positive column count, found ${x.columns}")
    val expected = checkedSize(x.rows, x.columns, what)
    requireThat(x.values.length == expected,
      s"$what values length ${x.values.length} does not match ${x.rows} x ${x.columns}")
    requireFinite(x.values, what)

  def requireFinite(values: Array[Double], what: String): Unit =
    values.indices.find(index => !java.lang.Double.isFinite(values(index))).foreach { index =>
      throw IllegalArgumentException(s"$what contains a nonfinite value at index $index")
    }

  def requireFinite(value: Double, what: String): Unit =
    requireThat(java.lang.Double.isFinite(value), s"$what must be finite")
