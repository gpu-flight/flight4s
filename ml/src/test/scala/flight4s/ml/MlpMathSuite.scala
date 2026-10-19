package flight4s.ml

import munit.FunSuite

class MlpMathSuite extends FunSuite:
  // Central differences with h = 1e-6 bound the comparison at about 1e-8 for these magnitudes.
  private val gradientTolerance = 1e-7

  private def parameters(): MlpMath.Parameters =
    MlpMath.Parameters.fromModel(MlpModel.initialized(MlpConfig(inputSize = 3, hiddenSize = 4, seed = 7L)))

  private def dataset(rows: Int, columns: Int, seed: Long): (DenseMatrix, Array[Double]) =
    val random = java.util.Random(seed)
    val x = DenseMatrix(rows, columns, Array.fill(rows * columns)(random.nextDouble() * 2.0 - 1.0))
    val y = Array.fill(rows)(random.nextDouble() * 2.0 - 1.0)
    (x, y)

  test("the forward pass matches an explicitly written two-by-two computation"):
    val model = MlpModel.fromParameters(2, 2,
      Array(0.5, -0.25, 0.1, 0.3), Array(0.1, -0.2), Array(1.5, -2.0), 0.25)
    val x = DenseMatrix(2, 2, Array(1.0, 2.0, -0.5, 0.0))
    val forward = MlpMath.forward(MlpMath.Parameters.fromModel(model), x)
    // The reference accumulates the bias first, then terms in index order; written identically here.
    val firstHidden = Array(math.tanh(0.1 + 1.0 * 0.5 + 2.0 * 0.1), math.tanh(-0.2 + 1.0 * -0.25 + 2.0 * 0.3))
    val secondHidden = Array(math.tanh(0.1 + -0.5 * 0.5 + 0.0 * 0.1), math.tanh(-0.2 + -0.5 * -0.25 + 0.0 * 0.3))
    assertEquals(forward.hidden.toVector, (firstHidden ++ secondHidden).toVector)
    assertEquals(forward.predictions.toVector, Vector(
      0.25 + 1.5 * firstHidden(0) + -2.0 * firstHidden(1),
      0.25 + 1.5 * secondHidden(0) + -2.0 * secondHidden(1)))

  test("mean squared error and the regularized objective match their definitions"):
    assertEquals(MlpMath.meanSquaredError(Array(1.0, 2.0), Array(0.0, 4.0)), 2.5)
    val fixed = MlpMath.Parameters(1, 1, Array(2.0), Array(5.0), Array(3.0), 7.0)
    // Biases are excluded: 2.5 + (0.1 / 2) * (4 + 9).
    assertEquals(MlpMath.objective(fixed, 2.5, 0.1), 2.5 + 0.05 * 13.0)

  test("loss reductions preserve sequential addition across widely different magnitudes"):
    val large = 134217728.0 // 2^27: its square is 2^54, where adding 1.0 rounds away.
    val predictions = Array(large) ++ Array.fill(8)(1.0)
    assertEquals(MlpMath.meanSquaredError(predictions, Array.fill(9)(0.0)), large * large / 9.0)

    val current = MlpMath.Parameters(1, 8,
      Array(large) ++ Array.fill(7)(0.0), Array.fill(8)(0.0), Array.fill(8)(1.0), 0.0)
    // Summing W2 separately would retain its 8.0 and change the reference result.
    assertEquals(MlpMath.objective(current, mse = 0.0, l2 = 2.0), large * large)

  private def objectiveOf(current: MlpMath.Parameters, x: DenseMatrix, y: Array[Double], l2: Double): Double =
    val forward = MlpMath.forward(current, x)
    MlpMath.objective(current, MlpMath.meanSquaredError(forward.predictions, y), l2)

  test("analytical gradients match central finite differences with and without L2"):
    val (x, y) = dataset(rows = 5, columns = 3, seed = 11L)
    for l2 <- Vector(0.0, 0.01) do
      val current = parameters()
      val forward = MlpMath.forward(current, x)
      val gradients = MlpMath.gradients(current, x, y, forward, l2)
      val h = 1e-6
      def numerical(read: () => Double, write: Double => Unit): Double =
        val original = read()
        write(original + h)
        val plus = objectiveOf(current, x, y, l2)
        write(original - h)
        val minus = objectiveOf(current, x, y, l2)
        write(original)
        (plus - minus) / (2.0 * h)
      for index <- current.w1.indices do
        assertEqualsDouble(gradients.w1(index),
          numerical(() => current.w1(index), value => current.w1(index) = value), gradientTolerance, s"w1($index) l2=$l2")
      for index <- current.b1.indices do
        assertEqualsDouble(gradients.b1(index),
          numerical(() => current.b1(index), value => current.b1(index) = value), gradientTolerance, s"b1($index) l2=$l2")
      for index <- current.w2.indices do
        assertEqualsDouble(gradients.w2(index),
          numerical(() => current.w2(index), value => current.w2(index) = value), gradientTolerance, s"w2($index) l2=$l2")
      assertEqualsDouble(gradients.b2,
        numerical(() => current.b2, value => current.b2 = value), gradientTolerance, s"b2 l2=$l2")

  test("L2 shifts weight gradients by lambda times the weight and leaves biases untouched"):
    val (x, y) = dataset(rows = 6, columns = 3, seed = 13L)
    val current = parameters()
    val forward = MlpMath.forward(current, x)
    val plain = MlpMath.gradients(current, x, y, forward, 0.0)
    val regularized = MlpMath.gradients(current, x, y, forward, 0.25)
    for index <- current.w1.indices do
      assertEqualsDouble(regularized.w1(index), plain.w1(index) + 0.25 * current.w1(index), 1e-12, s"w1($index)")
    for index <- current.w2.indices do
      assertEqualsDouble(regularized.w2(index), plain.w2(index) + 0.25 * current.w2(index), 1e-12, s"w2($index)")
    assertEquals(regularized.b1.toVector, plain.b1.toVector)
    assertEquals(regularized.b2, plain.b2)

  test("the first Adam update applies bias-corrected moments exactly"):
    val current = MlpMath.Parameters(1, 1, Array(1.0), Array(-0.5), Array(2.0), 0.25)
    val state = MlpMath.AdamState(1, 1)
    val config = AdamConfig(learningRate = 0.01)
    val gradients = MlpMath.Gradients(Array(0.3), Array(-0.7), Array(1.1), -0.2)
    def expected(parameter: Double, gradient: Double): Double =
      val m = (1.0 - config.beta1) * gradient
      val v = (1.0 - config.beta2) * gradient * gradient
      val mHat = m / (1.0 - config.beta1)
      val vHat = v / (1.0 - config.beta2)
      parameter - config.learningRate * mHat / (math.sqrt(vHat) + config.epsilon)
    MlpMath.adamStep(current, state, gradients, config)
    assertEquals(state.step, 1)
    assertEqualsDouble(current.w1(0), expected(1.0, 0.3), 1e-15, "w1")
    assertEqualsDouble(current.b1(0), expected(-0.5, -0.7), 1e-15, "b1")
    assertEqualsDouble(current.w2(0), expected(2.0, 1.1), 1e-15, "w2")
    assertEqualsDouble(current.b2, expected(0.25, -0.2), 1e-15, "b2")

  test("repeated Adam updates track an independent scalar reference implementation"):
    val current = MlpMath.Parameters(1, 1, Array(0.8), Array(0.0), Array(-1.2), 0.1)
    val state = MlpMath.AdamState(1, 1)
    val config = AdamConfig(learningRate = 0.02, beta1 = 0.8, beta2 = 0.95, epsilon = 1e-9)
    val gradientSteps = Vector(0.5, -0.25, 0.75, 0.1)
    var parameter = 0.8
    var m = 0.0
    var v = 0.0
    for (gradient, index) <- gradientSteps.zipWithIndex do
      MlpMath.adamStep(current, state, MlpMath.Gradients(Array(gradient), Array(0.0), Array(0.0), 0.0), config)
      m = config.beta1 * m + (1.0 - config.beta1) * gradient
      v = config.beta2 * v + (1.0 - config.beta2) * gradient * gradient
      val step = index + 1
      val mHat = m / (1.0 - math.pow(config.beta1, step))
      val vHat = v / (1.0 - math.pow(config.beta2, step))
      parameter -= config.learningRate * mHat / (math.sqrt(vHat) + config.epsilon)
      assertEquals(state.step, step)
      assertEqualsDouble(current.w1(0), parameter, 1e-15, s"step $step")
