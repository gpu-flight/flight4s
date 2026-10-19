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
    val hidden = new Array[Double](rows * hiddenSize)
    val predictions = new Array[Double](rows)
    var row = 0
    while row < rows do
      val rowBase = row * inputSize
      var output = parameters.b2
      var h = 0
      while h < hiddenSize do
        var z = parameters.b1(h)
        var d = 0
        while d < inputSize do
          z += x.values(rowBase + d) * parameters.w1(d * hiddenSize + h)
          d += 1
        val activation = math.tanh(z)
        hidden(row * hiddenSize + h) = activation
        output += activation * parameters.w2(h)
        h += 1
      predictions(row) = output
      row += 1
    Forward(hidden, predictions)

  def meanSquaredError(predictions: Array[Double], targets: Array[Double]): Double =
    var sum = 0.0
    var index = 0
    while index < predictions.length do
      val difference = predictions(index) - targets(index)
      sum += difference * difference
      index += 1
    sum / predictions.length

  /** objective = mse + (l2 / 2) * (sum W1^2 + sum W2^2); biases excluded. */
  def objective(parameters: Parameters, mse: Double, l2: Double): Double =
    if l2 == 0.0 then mse
    else
      var weights = 0.0
      var index = 0
      while index < parameters.w1.length do
        weights += parameters.w1(index) * parameters.w1(index)
        index += 1
      index = 0
      while index < parameters.w2.length do
        weights += parameters.w2(index) * parameters.w2(index)
        index += 1
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
    var row = 0
    while row < rows do
      val outputGradient = 2.0 * (forward.predictions(row) - targets(row)) / rows
      b2Gradient += outputGradient
      val hiddenBase = row * hiddenSize
      val rowBase = row * inputSize
      var h = 0
      while h < hiddenSize do
        val activation = forward.hidden(hiddenBase + h)
        w2Gradient(h) += activation * outputGradient
        val hiddenGradient = outputGradient * parameters.w2(h) * (1.0 - activation * activation)
        b1Gradient(h) += hiddenGradient
        var d = 0
        while d < inputSize do
          w1Gradient(d * hiddenSize + h) += x.values(rowBase + d) * hiddenGradient
          d += 1
        h += 1
      row += 1
    if l2 != 0.0 then
      var index = 0
      while index < w1Gradient.length do
        w1Gradient(index) += l2 * parameters.w1(index)
        index += 1
      var h = 0
      while h < hiddenSize do
        w2Gradient(h) += l2 * parameters.w2(h)
        h += 1
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
      var index = 0
      while index < values.length do
        val (value, nextM, nextV) = updated(values(index), gradientValues(index), m(index), v(index))
        values(index) = value
        m(index) = nextM
        v(index) = nextV
        index += 1
    updateArray(parameters.w1, gradients.w1, state.w1M, state.w1V)
    updateArray(parameters.b1, gradients.b1, state.b1M, state.b1V)
    updateArray(parameters.w2, gradients.w2, state.w2M, state.w2V)
    val (b2, b2M, b2V) = updated(parameters.b2, gradients.b2, state.b2M, state.b2V)
    parameters.b2 = b2
    state.b2M = b2M
    state.b2V = b2V
