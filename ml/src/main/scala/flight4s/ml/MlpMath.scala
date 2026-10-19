package flight4s.ml

/** CPU reference mathematics: the oracle for the GPU backend and the CPU inference path. */
private[ml] object MlpMath:
  final case class Forward(hidden: Array[Double], predictions: Array[Double])

  final case class Gradients(w1: Array[Double], b1: Array[Double], w2: Array[Double], b2: Double)

  /** Mutable Adam state; step counts completed updates. */
  final class AdamState(val inputSize: Int, val hiddenSize: Int):
    val w1M: Array[Double] = Array.fill(inputSize * hiddenSize)(0.0)
    val w1V: Array[Double] = Array.fill(inputSize * hiddenSize)(0.0)
    val b1M: Array[Double] = Array.fill(hiddenSize)(0.0)
    val b1V: Array[Double] = Array.fill(hiddenSize)(0.0)
    val w2M: Array[Double] = Array.fill(hiddenSize)(0.0)
    val w2V: Array[Double] = Array.fill(hiddenSize)(0.0)
    var b2M: Double = 0.0
    var b2V: Double = 0.0
    var step: Int = 0

  /** Mutable parameters owned by a trainer; snapshots copy in and out. */
  final class Parameters(val inputSize: Int, val hiddenSize: Int,
      val w1: Array[Double], val b1: Array[Double], val w2: Array[Double], var b2: Double):
    def toModel: MlpModel = MlpModel(inputSize, hiddenSize, w1.clone(), b1.clone(), w2.clone(), b2)

  object Parameters:
    def fromModel(model: MlpModel): Parameters =
      Parameters(model.inputSize, model.hiddenSize, model.w1.clone(), model.b1.clone(), model.w2.clone(), model.b2)

  def forward(parameters: Parameters, x: DenseMatrix): Forward =
    val rows = x.rows
    val inputSize = parameters.inputSize
    val hiddenSize = parameters.hiddenSize
    val inputIndices = 0 until inputSize
    val hiddenIndices = 0 until hiddenSize

    val hidden = Array.tabulate(rows * hiddenSize) { index =>
      val rowBase = (index / hiddenSize) * inputSize
      val h = index % hiddenSize
      val z = inputIndices.foldLeft(parameters.b1(h)) { (sum, d) =>
        sum + x.values(rowBase + d) * parameters.w1(d * hiddenSize + h)
      }
      math.tanh(z)
    }

    val predictions = Array.tabulate(rows) { row =>
      val hiddenBase = row * hiddenSize
      hiddenIndices.foldLeft(parameters.b2) { (sum, h) =>
        sum + hidden(hiddenBase + h) * parameters.w2(h)
      }
    }
    Forward(hidden, predictions)

  def meanSquaredError(predictions: Array[Double], targets: Array[Double]): Double =
    val sum = predictions.indices.foldLeft(0.0) { (sum, index) =>
      val difference = predictions(index) - targets(index)
      sum + difference * difference
    }
    sum / predictions.length

  /** objective = mse + (l2 / 2) * (sum W1^2 + sum W2^2); biases excluded. */
  def objective(parameters: Parameters, mse: Double, l2: Double): Double =
    if l2 == 0.0 then mse
    else
      val inputWeights = parameters.w1.foldLeft(0.0) { (sum, weight) =>
        sum + weight * weight
      }
      // Continue the same accumulator to preserve W1-then-W2 addition order.
      val weights = parameters.w2.foldLeft(inputWeights) { (sum, weight) =>
        sum + weight * weight
      }
      mse + (l2 / 2.0) * weights

  /** Analytical full-batch gradients of the objective; L2 flows through the weight gradients. */
  def gradients(parameters: Parameters, x: DenseMatrix, targets: Array[Double],
      forward: Forward, l2: Double): Gradients =
    val rows = x.rows
    val inputSize = parameters.inputSize
    val hiddenSize = parameters.hiddenSize
    val w1Gradient = new Array[Double](inputSize * hiddenSize)
    val b1Gradient = new Array[Double](hiddenSize)
    val w2Gradient = new Array[Double](hiddenSize)
    var b2Gradient = 0.0
    for row <- 0 until rows do
      val outputGradient = 2.0 * (forward.predictions(row) - targets(row)) / rows
      b2Gradient += outputGradient
      val hiddenBase = row * hiddenSize
      val rowBase = row * inputSize
      for h <- 0 until hiddenSize do
        val activation = forward.hidden(hiddenBase + h)
        w2Gradient(h) += activation * outputGradient
        val hiddenGradient = outputGradient * parameters.w2(h) * (1.0 - activation * activation)
        b1Gradient(h) += hiddenGradient
        for d <- 0 until inputSize do
          w1Gradient(d * hiddenSize + h) += x.values(rowBase + d) * hiddenGradient
    if l2 != 0.0 then
      for index <- w1Gradient.indices do
        w1Gradient(index) += l2 * parameters.w1(index)
      for h <- w2Gradient.indices do
        w2Gradient(h) += l2 * parameters.w2(h)
    Gradients(w1Gradient, b1Gradient, w2Gradient, b2Gradient)

  /** One Adam update with bias correction for both moments; increments state.step once. */
  def adamStep(parameters: Parameters, state: AdamState, gradients: Gradients, config: AdamConfig): Unit =
    state.step += 1
    val mCorrection = 1.0 - math.pow(config.beta1, state.step)
    val vCorrection = 1.0 - math.pow(config.beta2, state.step)
    def updated(value: Double, gradient: Double, m: Double, v: Double): (Double, Double, Double) =
      val nextM = config.beta1 * m + (1.0 - config.beta1) * gradient
      val nextV = config.beta2 * v + (1.0 - config.beta2) * gradient * gradient
      val step = config.learningRate * (nextM / mCorrection) / (math.sqrt(nextV / vCorrection) + config.epsilon)
      (value - step, nextM, nextV)
    def updateArray(values: Array[Double], gradientValues: Array[Double],
        m: Array[Double], v: Array[Double]): Unit =
      for index <- values.indices do
        val (value, nextM, nextV) = updated(values(index), gradientValues(index), m(index), v(index))
        values(index) = value
        m(index) = nextM
        v(index) = nextV
    updateArray(parameters.w1, gradients.w1, state.w1M, state.w1V)
    updateArray(parameters.b1, gradients.b1, state.b1M, state.b1V)
    updateArray(parameters.w2, gradients.w2, state.w2M, state.w2V)
    val (b2, b2M, b2V) = updated(parameters.b2, gradients.b2, state.b2M, state.b2V)
    parameters.b2 = b2
    state.b2M = b2M
    state.b2V = b2V
