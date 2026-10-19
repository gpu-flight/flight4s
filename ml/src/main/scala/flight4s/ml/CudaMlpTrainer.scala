package flight4s.ml

import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.runtime.cuda.{CudaContext, CudaDeviceBuffer, CudaFunction, CudaModule, NvrtcCompiler}

/** CUDA-backed trainer: forward propagation, backpropagation, gradient aggregation and Adam
  * updates all run on the GPU over one flat device-resident parameter buffer. Only the
  * predictions (and the weight-square scalar when L2 is active) come back per epoch; the
  * dataset and parameters stay on the device. Metrics are measured on a fresh forward pass
  * after the update, and loadModel resets the Adam moments and step count. */
final class CudaMlpTrainer private (
    config: MlpConfig,
    optimizer: AdamConfig,
    context: CudaContext,
    module: CudaModule,
    kernels: MlpKernels.GeneratedKernels,
    predictFunction: CudaFunction[MlpKernels.PredictArgs],
    trainingForwardFunction: CudaFunction[MlpKernels.TrainingForwardArgs],
    gradientsFunction: CudaFunction[MlpKernels.GradientArgs],
    adamFunction: CudaFunction[MlpKernels.AdamArgs],
    squaresFunction: CudaFunction[MlpKernels.SquaresArgs],
    parametersBuffer: CudaDeviceBuffer[Double],
    gradientsBuffer: CudaDeviceBuffer[Double],
    moments1Buffer: CudaDeviceBuffer[Double],
    moments2Buffer: CudaDeviceBuffer[Double],
    squaresBuffer: CudaDeviceBuffer[Double]
) extends MlpTrainer:
  private val totalParameters = MlpKernels.parameterCount(config.inputSize, config.hiddenSize)
  private val w1Count = config.inputSize * config.hiddenSize
  private var adamStep = 0
  private var epochs = 0
  private var closed = false
  private var batches = List.empty[CudaBatch]

  private final class CudaBatch(
      val xBuffer: CudaDeviceBuffer[Double],
      val yBuffer: CudaDeviceBuffer[Double],
      val hiddenBuffer: CudaDeviceBuffer[Double],
      val predictionsBuffer: CudaDeviceBuffer[Double],
      val targets: Array[Double],
      val rows: Int,
      val columns: Int
  ) extends TrainingBatch:
    private[CudaMlpTrainer] var batchClosed = false
    private[CudaMlpTrainer] val owner: CudaMlpTrainer = CudaMlpTrainer.this
    def close(): Unit =
      if !batchClosed then
        batchClosed = true
        xBuffer.close()
        yBuffer.close()
        hiddenBuffer.close()
        predictionsBuffer.close()

  private def requireOpen(): Unit =
    MlValidation.requireThat(!closed, "this trainer is closed")

  private def run[A](what: String)(result: Either[?, A]): A = result match
    case Right(value) => value
    case Left(failure) => throw IllegalStateException(s"$what failed: $failure")

  private def allocate(count: Int, what: String): CudaDeviceBuffer[Double] =
    run(s"allocating $what")(context.allocate[Double](count))

  private def gridFor(count: Int): LaunchConfig =
    LaunchConfig(Grid.x(math.max(1, (count + 127) / 128)), LaunchBlock.x(128))

  private def flatten(model: MlpModel): Array[Double] =
    val flat = new Array[Double](totalParameters)
    System.arraycopy(model.w1, 0, flat, 0, w1Count)
    System.arraycopy(model.b1, 0, flat, w1Count, config.hiddenSize)
    System.arraycopy(model.w2, 0, flat, w1Count + config.hiddenSize, config.hiddenSize)
    flat(totalParameters - 1) = model.b2
    flat

  private[ml] def installParameters(model: MlpModel): Unit =
    run("uploading the parameters")(parametersBuffer.copyFrom(flatten(model)))
    val zeros = new Array[Double](totalParameters)
    run("resetting the first moments")(moments1Buffer.copyFrom(zeros))
    run("resetting the second moments")(moments2Buffer.copyFrom(zeros))
    adamStep = 0

  def upload(x: DenseMatrix, y: Array[Double]): TrainingBatch =
    requireOpen()
    MlValidation.validateMatrix(x, "x")
    MlValidation.requireThat(x.columns == config.inputSize,
      s"x has ${x.columns} columns but the model expects ${config.inputSize}")
    MlValidation.requireThat(y.length == x.rows, s"y length ${y.length} does not match ${x.rows} rows")
    MlValidation.requireFinite(y, "y")
    var allocated = List.empty[CudaDeviceBuffer[Double]]
    try
      def tracked(count: Int, what: String): CudaDeviceBuffer[Double] =
        val buffer = allocate(count, what)
        allocated = buffer :: allocated
        buffer
      val xBuffer = tracked(x.values.length, "the training features")
      val yBuffer = tracked(y.length, "the training targets")
      val hiddenBuffer = tracked(MlValidation.checkedSize(x.rows, config.hiddenSize, "hidden activations"),
        "the hidden activations")
      val predictionsBuffer = tracked(x.rows, "the training predictions")
      run("uploading the training features")(xBuffer.copyFrom(x.values))
      run("uploading the training targets")(yBuffer.copyFrom(y))
      val batch = CudaBatch(xBuffer, yBuffer, hiddenBuffer, predictionsBuffer, y.clone(), x.rows, x.columns)
      batches = batch :: batches
      batch
    catch
      case error: Throwable =>
        allocated.foreach(_.close())
        throw error

  def trainEpoch(batch: TrainingBatch): TrainingMetrics =
    requireOpen()
    val owned = batch match
      case cuda: CudaBatch if cuda.owner eq this => cuda
      case _ => throw IllegalArgumentException("this batch belongs to a different trainer")
    MlValidation.requireThat(!owned.batchClosed, "this batch is closed")
    def forward(): Unit =
      run("launching the training forward kernel")(trainingForwardFunction.launch(
        kernels.trainingForwardDefinition.bind((parametersBuffer, owned.xBuffer, owned.hiddenBuffer,
          owned.predictionsBuffer, owned.rows, config.inputSize, config.hiddenSize)), gridFor(owned.rows)))
    forward()
    run("launching the gradient kernel")(gradientsFunction.launch(
      kernels.gradientsDefinition.bind((parametersBuffer, owned.xBuffer, owned.yBuffer, owned.hiddenBuffer,
        owned.predictionsBuffer, gradientsBuffer, owned.rows.toDouble, optimizer.l2,
        owned.rows, config.inputSize, config.hiddenSize)), gridFor(totalParameters)))
    adamStep += 1
    val moment1Correction = 1.0 - math.pow(optimizer.beta1, adamStep)
    val moment2Correction = 1.0 - math.pow(optimizer.beta2, adamStep)
    run("launching the Adam kernel")(adamFunction.launch(
      kernels.adamDefinition.bind((parametersBuffer, gradientsBuffer, moments1Buffer, moments2Buffer,
        optimizer.learningRate, optimizer.beta1, optimizer.beta2, optimizer.epsilon,
        moment1Correction, moment2Correction, totalParameters)), gridFor(totalParameters)))
    forward()
    run("synchronizing the training epoch")(context.synchronize())
    val predictions = run("downloading the post-update predictions")(owned.predictionsBuffer.copyToArray())
    val mse = MlpMath.meanSquaredError(predictions, owned.targets)
    val objective =
      if optimizer.l2 == 0.0 then mse
      else
        run("launching the weight-square kernel")(squaresFunction.launch(
          kernels.squaresDefinition.bind((parametersBuffer, squaresBuffer, config.inputSize, config.hiddenSize)),
          LaunchConfig(Grid.x(1), LaunchBlock.x(1))))
        run("synchronizing the weight squares")(context.synchronize())
        val squares = run("downloading the weight squares")(squaresBuffer.copyToArray())
        mse + (optimizer.l2 / 2.0) * squares(0)
    epochs += 1
    TrainingMetrics(epochs, mse, objective)

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
        run("launching the forward kernel")(predictFunction.launch(
          kernels.predictDefinition.bind((parametersBuffer, xBuffer, predictions,
            x.rows, config.inputSize, config.hiddenSize)), gridFor(x.rows)))
        run("synchronizing the forward kernel")(context.synchronize())
        run("downloading the predictions")(predictions.copyToArray())
      finally predictions.close()
    finally xBuffer.close()

  def snapshotModel(): MlpModel =
    requireOpen()
    run("synchronizing before the snapshot")(context.synchronize())
    val flat = run("downloading the parameters")(parametersBuffer.copyToArray())
    val w1 = java.util.Arrays.copyOfRange(flat, 0, w1Count)
    val b1 = java.util.Arrays.copyOfRange(flat, w1Count, w1Count + config.hiddenSize)
    val w2 = java.util.Arrays.copyOfRange(flat, w1Count + config.hiddenSize, w1Count + 2 * config.hiddenSize)
    MlpModel(config.inputSize, config.hiddenSize, w1, b1, w2, flat(totalParameters - 1))

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
      squaresBuffer.close()
      moments2Buffer.close()
      moments1Buffer.close()
      gradientsBuffer.close()
      parametersBuffer.close()
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
      val kernels = MlpKernels.generateAll() match
        case Right(value) => value
        case Left(message) => throw IllegalStateException(s"generating the ML kernels failed: $message")
      val artifact = NvrtcCompiler.compile(kernels.module, context.computeCapability, "flight4s_ml.cu") match
        case Right(value) => value
        case Left(failure) => throw IllegalStateException(
          s"compiling the ML kernels failed: ${failure.message}\n${failure.compileLog}")
      val module = run("loading the ML kernel module")(context.load(artifact))
      val totalParameters = MlpKernels.parameterCount(config.inputSize, config.hiddenSize)
      val trainer = new CudaMlpTrainer(config, optimizer, context, module, kernels,
        run("resolving the predict kernel")(module.function(kernels.predict)),
        run("resolving the training forward kernel")(module.function(kernels.trainingForward)),
        run("resolving the gradient kernel")(module.function(kernels.gradients)),
        run("resolving the Adam kernel")(module.function(kernels.adam)),
        run("resolving the weight-square kernel")(module.function(kernels.squares)),
        run("allocating the parameters")(context.allocate[Double](totalParameters)),
        run("allocating the gradients")(context.allocate[Double](totalParameters)),
        run("allocating the first moments")(context.allocate[Double](totalParameters)),
        run("allocating the second moments")(context.allocate[Double](totalParameters)),
        run("allocating the weight-square scratch")(context.allocate[Double](1)))
      trainer.installParameters(MlpModel.initialized(config))
      constructed = trainer
      trainer
    finally
      // Closing the context also releases any resources it already registered.
      if constructed == null then context.close()
