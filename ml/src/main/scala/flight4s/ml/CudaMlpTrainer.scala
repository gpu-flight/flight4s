package flight4s.ml

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}
import flight4s.core.ir.Kernel
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, CudaDeviceBuffer, CudaFunction, CudaModule, NvrtcCompiler}

/** CUDA-backed trainer; this slice implements upload, GPU forward prediction, snapshots and
  * lifecycle. GPU training (backpropagation and Adam updates) lands in the next slice and
  * trainEpoch reports that honestly instead of falling back to the CPU. Parameters stay
  * resident in device buffers and are re-uploaded only when they change. */
final class CudaMlpTrainer private (
    config: MlpConfig,
    optimizer: AdamConfig,
    definition: Kernel[MlpKernels.ForwardArgs],
    context: CudaContext,
    module: CudaModule,
    forwardFunction: CudaFunction[MlpKernels.ForwardArgs],
    w1Buffer: CudaDeviceBuffer[Double],
    b1Buffer: CudaDeviceBuffer[Double],
    w2Buffer: CudaDeviceBuffer[Double]
) extends MlpTrainer:
  private var parameters = MlpMath.Parameters.fromModel(MlpModel.initialized(config))
  private var closed = false
  private var batches = List.empty[CudaBatch]

  private final class CudaBatch(val xBuffer: CudaDeviceBuffer[Double], val yBuffer: CudaDeviceBuffer[Double],
      val rows: Int, val columns: Int) extends TrainingBatch:
    private[CudaMlpTrainer] var batchClosed = false
    private[CudaMlpTrainer] val owner: CudaMlpTrainer = CudaMlpTrainer.this
    def close(): Unit =
      if !batchClosed then
        batchClosed = true
        xBuffer.close()
        yBuffer.close()

  private def requireOpen(): Unit =
    MlValidation.requireThat(!closed, "this trainer is closed")

  private def run[A](what: String)(result: Either[?, A]): A = result match
    case Right(value) => value
    case Left(failure) => throw IllegalStateException(s"$what failed: $failure")

  private def allocate(count: Int, what: String): CudaDeviceBuffer[Double] =
    run(s"allocating $what")(context.allocate[Double](count))

  private[ml] def installParameters(model: MlpModel): Unit =
    parameters = MlpMath.Parameters.fromModel(model)
    run("uploading w1")(w1Buffer.copyFrom(parameters.w1))
    run("uploading b1")(b1Buffer.copyFrom(parameters.b1))
    run("uploading w2")(w2Buffer.copyFrom(parameters.w2))

  def upload(x: DenseMatrix, y: Array[Double]): TrainingBatch =
    requireOpen()
    MlValidation.validateMatrix(x, "x")
    MlValidation.requireThat(x.columns == config.inputSize,
      s"x has ${x.columns} columns but the model expects ${config.inputSize}")
    MlValidation.requireThat(y.length == x.rows, s"y length ${y.length} does not match ${x.rows} rows")
    MlValidation.requireFinite(y, "y")
    val xBuffer = allocate(x.values.length, "the training features")
    val yBuffer =
      try allocate(y.length, "the training targets")
      catch
        case error: Throwable =>
          xBuffer.close()
          throw error
    try
      run("uploading the training features")(xBuffer.copyFrom(x.values))
      run("uploading the training targets")(yBuffer.copyFrom(y))
    catch
      case error: Throwable =>
        xBuffer.close()
        yBuffer.close()
        throw error
    val batch = CudaBatch(xBuffer, yBuffer, x.rows, x.columns)
    batches = batch :: batches
    batch

  def trainEpoch(batch: TrainingBatch): TrainingMetrics =
    requireOpen()
    val owned = batch match
      case cuda: CudaBatch if cuda.owner eq this => cuda
      case _ => throw IllegalArgumentException("this batch belongs to a different trainer")
    MlValidation.requireThat(!owned.batchClosed, "this batch is closed")
    throw UnsupportedOperationException(
      "GPU training is not implemented yet; the backpropagation and Adam slice follows this one")

  def predict(x: DenseMatrix): Array[Double] =
    requireOpen()
    MlValidation.validateMatrix(x, "x")
    MlValidation.requireThat(x.columns == config.inputSize,
      s"x has ${x.columns} columns but the model expects ${config.inputSize}")
    val xBuffer = allocate(x.values.length, "the prediction features")
    try
      val predictions = allocate(x.rows, "the predictions")
      try
        run("uploading the prediction features")(xBuffer.copyFrom(x.values))
        val launch = LaunchConfig(Grid.x(math.max(1, (x.rows + 127) / 128)), LaunchBlock.x(128))
        run("launching the forward kernel")(forwardFunction.launch(definition.bind((
          xBuffer, w1Buffer, b1Buffer, w2Buffer, parameters.b2, predictions,
          x.rows, config.inputSize, config.hiddenSize)), launch))
        run("synchronizing the forward kernel")(context.synchronize())
        run("downloading the predictions")(predictions.copyToArray())
      finally predictions.close()
    finally xBuffer.close()

  def snapshotModel(): MlpModel =
    requireOpen()
    parameters.toModel

  def loadModel(model: MlpModel): Unit =
    requireOpen()
    MlpModel.validate(model)
    MlValidation.requireThat(model.inputSize == config.inputSize && model.hiddenSize == config.hiddenSize,
      s"model ${model.inputSize} x ${model.hiddenSize} does not match trainer ${config.inputSize} x ${config.hiddenSize}")
    installParameters(model)

  def executionInfo: ExecutionInfo =
    val capability = context.computeCapability
    ExecutionInfo(MlBackend.Cuda,
      s"CUDA device ${context.deviceOrdinal} (sm_${capability.major}${capability.minor})",
      MlpModel.Precision, "NVRTC-compiled flight4s explicit-DSL kernels")

  def close(): Unit =
    if !closed then
      closed = true
      batches.foreach(_.close())
      batches = Nil
      w1Buffer.close()
      b1Buffer.close()
      w2Buffer.close()
      module.close()
      context.close()

object CudaMlpTrainer:
  /** Opens the CUDA device and compiles the module's kernels; fails with an actionable error
    * when CUDA is unavailable rather than silently running on the CPU. */
  def apply(config: MlpConfig, optimizer: AdamConfig = AdamConfig(), deviceOrdinal: Int = 0): CudaMlpTrainer =
    MlpConfig.validate(config)
    AdamConfig.validate(optimizer)
    val context =
      try
        CudaContext.open(deviceOrdinal) match
          case Right(opened) => opened
          case Left(failure) => throw IllegalStateException(
            s"CUDA is unavailable on device $deviceOrdinal: ${failure.message}. Install a CUDA driver " +
              "and pass -Dflight4s.cuda.native.path pointing at the flight4s_cuda native library.")
      catch
        case error: LinkageError => throw IllegalStateException(
          "the flight4s CUDA native library is not loaded; pass -Dflight4s.cuda.native.path " +
            "pointing at the flight4s_cuda native library", error)
    var constructed: CudaMlpTrainer = null
    try
      def run[A](what: String)(result: Either[?, A]): A = result match
        case Right(value) => value
        case Left(failure) => throw IllegalStateException(s"$what failed: $failure")
      val definition = MlpKernels.forward
      val generated = CudaCodegen.generate(definition) match
        case Right(value) => value
        case Left(error) => throw IllegalStateException(s"generating the forward kernel failed: ${error.message}")
      val artifact = NvrtcCompiler.compile(
        GeneratedCudaModule(generated.cudaSource, generated.sourceMap, generated.compilerOptions, Vector(generated)),
        context.computeCapability, "flight4s_ml.cu") match
        case Right(value) => value
        case Left(failure) => throw IllegalStateException(
          s"compiling the forward kernel failed: ${failure.message}\n${failure.compileLog}")
      val module = run("loading the forward kernel module")(context.load(artifact))
      val function = run("resolving the forward kernel")(module.function(generated))
      val w1Buffer = run("allocating w1")(context.allocate[Double](config.inputSize * config.hiddenSize))
      val b1Buffer = run("allocating b1")(context.allocate[Double](config.hiddenSize))
      val w2Buffer = run("allocating w2")(context.allocate[Double](config.hiddenSize))
      val trainer = new CudaMlpTrainer(config, optimizer, definition, context, module, function,
        w1Buffer, b1Buffer, w2Buffer)
      trainer.installParameters(MlpModel.initialized(config))
      constructed = trainer
      trainer
    finally
      // Closing the context also releases any resources it already registered.
      if constructed == null then context.close()
