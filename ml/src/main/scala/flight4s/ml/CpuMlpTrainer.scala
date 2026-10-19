package flight4s.ml

/** CPU inference from a model snapshot; needs no CUDA runtime or native library. */
object CpuInference:
  def predict(model: MlpModel, x: DenseMatrix): Array[Double] =
    MlpModel.validate(model)
    MlValidation.validateMatrix(x, "x")
    MlValidation.requireThat(x.columns == model.inputSize,
      s"x has ${x.columns} columns but the model expects ${model.inputSize}")
    MlpMath.forward(MlpMath.Parameters.fromModel(model), x).predictions

/** Explicit CPU-backend trainer sharing the reference mathematics; also the oracle for CUDA tests. */
final class CpuMlpTrainer private (config: MlpConfig, optimizer: AdamConfig) extends MlpTrainer:
  private var parameters = MlpMath.Parameters.fromModel(MlpModel.initialized(config))
  private var adam = MlpMath.AdamState(config.inputSize, config.hiddenSize)
  private var epochs = 0
  private var closed = false

  private final class CpuBatch(val x: DenseMatrix, val y: Array[Double]) extends TrainingBatch:
    private[CpuMlpTrainer] var batchClosed = false
    private[CpuMlpTrainer] val owner: CpuMlpTrainer = CpuMlpTrainer.this
    def rows: Int = x.rows
    def columns: Int = x.columns
    def close(): Unit = batchClosed = true

  private def requireOpen(): Unit =
    MlValidation.requireThat(!closed, "this trainer is closed")

  def upload(x: DenseMatrix, y: Array[Double]): TrainingBatch =
    requireOpen()
    MlValidation.validateMatrix(x, "x")
    MlValidation.requireThat(x.columns == config.inputSize,
      s"x has ${x.columns} columns but the model expects ${config.inputSize}")
    MlValidation.requireThat(y.length == x.rows, s"y length ${y.length} does not match ${x.rows} rows")
    MlValidation.requireFinite(y, "y")
    // Copies keep ownership inside the batch; later caller mutation cannot change training data.
    CpuBatch(DenseMatrix(x.rows, x.columns, x.values.clone()), y.clone())

  def trainEpoch(batch: TrainingBatch): TrainingMetrics =
    requireOpen()
    val owned = batch match
      case cpu: CpuBatch if cpu.owner eq this => cpu
      case _ => throw IllegalArgumentException("this batch belongs to a different trainer")
    MlValidation.requireThat(!owned.batchClosed, "this batch is closed")
    val before = MlpMath.forward(parameters, owned.x)
    val gradients = MlpMath.gradients(parameters, owned.x, owned.y, before, optimizer.l2)
    MlpMath.adamStep(parameters, adam, gradients, optimizer)
    epochs += 1
    // Metrics are measured on a fresh forward pass after the update.
    val after = MlpMath.forward(parameters, owned.x)
    val mse = MlpMath.meanSquaredError(after.predictions, owned.y)
    TrainingMetrics(epochs, mse, MlpMath.objective(parameters, mse, optimizer.l2))

  def predict(x: DenseMatrix): Array[Double] =
    requireOpen()
    MlValidation.validateMatrix(x, "x")
    MlValidation.requireThat(x.columns == config.inputSize,
      s"x has ${x.columns} columns but the model expects ${config.inputSize}")
    MlpMath.forward(parameters, x).predictions

  def snapshotModel(): MlpModel =
    requireOpen()
    parameters.toModel

  def loadModel(model: MlpModel): Unit =
    requireOpen()
    MlpModel.validate(model)
    MlValidation.requireThat(model.inputSize == config.inputSize && model.hiddenSize == config.hiddenSize,
      s"model ${model.inputSize} x ${model.hiddenSize} does not match trainer ${config.inputSize} x ${config.hiddenSize}")
    // Loading installs parameters and resets Adam moments and the step count.
    parameters = MlpMath.Parameters.fromModel(model)
    adam = MlpMath.AdamState(config.inputSize, config.hiddenSize)

  def executionInfo: ExecutionInfo =
    ExecutionInfo(MlBackend.Cpu, "JVM", MlpModel.Precision, s"Scala ${util.Properties.versionNumberString}")

  def close(): Unit = closed = true

object CpuMlpTrainer:
  def apply(config: MlpConfig, optimizer: AdamConfig = AdamConfig()): CpuMlpTrainer =
    MlpConfig.validate(config)
    AdamConfig.validate(optimizer)
    new CpuMlpTrainer(config, optimizer)
