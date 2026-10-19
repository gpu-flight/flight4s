package flight4s.ml

/** Architecture and seeding for the single-hidden-layer regression MLP. */
final case class MlpConfig(inputSize: Int, hiddenSize: Int = 8, seed: Long = 42L)

object MlpConfig:
  def validate(config: MlpConfig): Unit =
    MlValidation.requireThat(config.inputSize > 0, s"inputSize must be positive, found ${config.inputSize}")
    MlValidation.requireThat(config.hiddenSize > 0, s"hiddenSize must be positive, found ${config.hiddenSize}")
    MlValidation.checkedSize(config.inputSize, config.hiddenSize, "weight matrix")
    ()

/** Adam settings; l2 is coupled regularization applied through the gradient, never AdamW decay. */
final case class AdamConfig(
    learningRate: Double = 0.01,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    epsilon: Double = 1e-8,
    l2: Double = 0.0
)

object AdamConfig:
  def validate(config: AdamConfig): Unit =
    MlValidation.requireThat(config.learningRate > 0 && java.lang.Double.isFinite(config.learningRate),
      s"learningRate must be positive and finite, found ${config.learningRate}")
    MlValidation.requireThat(config.beta1 >= 0 && config.beta1 < 1, s"beta1 must be in [0, 1), found ${config.beta1}")
    MlValidation.requireThat(config.beta2 >= 0 && config.beta2 < 1, s"beta2 must be in [0, 1), found ${config.beta2}")
    MlValidation.requireThat(config.epsilon > 0 && java.lang.Double.isFinite(config.epsilon),
      s"epsilon must be positive and finite, found ${config.epsilon}")
    MlValidation.requireThat(config.l2 >= 0 && java.lang.Double.isFinite(config.l2),
      s"l2 must be nonnegative and finite, found ${config.l2}")

/** Metrics from one full-batch epoch, measured on a fresh forward pass after the update. */
final case class TrainingMetrics(epoch: Int, mse: Double, objective: Double)

enum MlBackend:
  case Cpu, Cuda

/** What actually executed the work; Cuda is never reported unless CUDA really ran. */
final case class ExecutionInfo(backend: MlBackend, deviceName: String, precision: String, details: String)
