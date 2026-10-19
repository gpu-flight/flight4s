package flight4s.ml

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}

/** Immutable snapshot of architecture and parameters; w1 is row-major inputSize x hiddenSize.
  * Arrays are owned by the snapshot: constructors copy, and accessors never expose live trainer state. */
final case class MlpModel(
    inputSize: Int,
    hiddenSize: Int,
    w1: Array[Double],
    b1: Array[Double],
    w2: Array[Double],
    b2: Double
)

object MlpModel:
  val FormatName = "flight4s-ml-mlp"
  val FormatVersion = 1
  val Activation = "tanh"
  val Output = "linear"
  val Precision = "float64"

  /** Xavier uniform over java.util.Random(seed): W1 then W2 drawn in index order, biases zero.
    * java.util.Random is algorithmically specified, so initialization is reproducible across platforms. */
  def initialized(config: MlpConfig): MlpModel =
    MlpConfig.validate(config)
    val random = java.util.Random(config.seed)
    def xavier(count: Int, fanIn: Int, fanOut: Int): Array[Double] =
      val limit = math.sqrt(6.0 / (fanIn + fanOut))
      Array.fill(count)((random.nextDouble() * 2.0 - 1.0) * limit)
    val w1 = xavier(config.inputSize * config.hiddenSize, config.inputSize, config.hiddenSize)
    val w2 = xavier(config.hiddenSize, config.hiddenSize, 1)
    MlpModel(config.inputSize, config.hiddenSize, w1, Array.fill(config.hiddenSize)(0.0), w2, 0.0)

  /** Validates and defensively copies caller-supplied parameters. */
  def fromParameters(inputSize: Int, hiddenSize: Int, w1: Array[Double], b1: Array[Double],
      w2: Array[Double], b2: Double): MlpModel =
    val model = MlpModel(inputSize, hiddenSize, w1.clone(), b1.clone(), w2.clone(), b2)
    validate(model)
    model

  def validate(model: MlpModel): Unit =
    MlpConfig.validate(MlpConfig(model.inputSize, model.hiddenSize))
    MlValidation.requireThat(model.w1.length == model.inputSize * model.hiddenSize,
      s"w1 length ${model.w1.length} does not match ${model.inputSize} x ${model.hiddenSize}")
    MlValidation.requireThat(model.b1.length == model.hiddenSize,
      s"b1 length ${model.b1.length} does not match hiddenSize ${model.hiddenSize}")
    MlValidation.requireThat(model.w2.length == model.hiddenSize,
      s"w2 length ${model.w2.length} does not match hiddenSize ${model.hiddenSize}")
    MlValidation.requireFinite(model.w1, "w1")
    MlValidation.requireFinite(model.b1, "b1")
    MlValidation.requireFinite(model.w2, "w2")
    MlValidation.requireFinite(model.b2, "b2")

  def toJson(model: MlpModel): String = MlpJson.write(model)

  def fromJson(text: String): MlpModel = MlpJson.read(text)

  def save(model: MlpModel, path: Path): Unit =
    validate(model)
    Files.write(path, toJson(model).getBytes(UTF_8))
    ()

  def load(path: Path): MlpModel = fromJson(String(Files.readAllBytes(path), UTF_8))
