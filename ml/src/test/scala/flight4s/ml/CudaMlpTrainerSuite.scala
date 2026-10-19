package flight4s.ml

import munit.FunSuite

class CudaMlpTrainerSuite extends FunSuite:
  // CUDA tanh and JVM Math.tanh may differ in the last ulps; all comparisons use this tolerance.
  private val tolerance = 1e-10

  private def assumeGpu(): Unit =
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run CUDA tests")

  private def randomMatrix(rows: Int, columns: Int, seed: Long): DenseMatrix =
    val random = java.util.Random(seed)
    DenseMatrix(rows, columns, Array.fill(rows * columns)(random.nextDouble() * 2.0 - 1.0))

  private def assertClose(actual: Array[Double], expected: Array[Double], clue: String): Unit =
    assertEquals(actual.length, expected.length, clue)
    for index <- actual.indices do
      assertEqualsDouble(actual(index), expected(index), tolerance, s"$clue index $index")

  test("GPU predictions match the CPU reference across consumer widths and edge shapes"):
    assumeGpu()
    for (rows, columns, hidden) <- Vector((1, 1, 1), (1, 5, 8), (3, 1, 2), (32, 3, 4),
        (513, 14, 8), (64, 18, 8), (65, 21, 8)) do
      val trainer = CudaMlpTrainer(MlpConfig(inputSize = columns, hiddenSize = hidden, seed = rows * 7L + columns))
      try
        val x = randomMatrix(rows, columns, seed = rows * 13L + columns)
        val expected = CpuInference.predict(trainer.snapshotModel(), x)
        assertClose(trainer.predict(x), expected, s"$rows x $columns h$hidden")
      finally trainer.close()

  test("explicitly loaded parameters reproduce CPU inference on the GPU"):
    assumeGpu()
    val model = MlpModel.fromParameters(2, 2,
      Array(0.5, -0.25, 0.1, 0.3), Array(0.1, -0.2), Array(1.5, -2.0), 0.25)
    val trainer = CudaMlpTrainer(MlpConfig(inputSize = 2, hiddenSize = 2))
    try
      trainer.loadModel(model)
      val x = DenseMatrix(2, 2, Array(1.0, 2.0, -0.5, 0.0))
      assertClose(trainer.predict(x), CpuInference.predict(model, x), "loaded model")
      val snapshot = trainer.snapshotModel()
      assertEquals(snapshot.w1.toVector, model.w1.toVector)
      assertEquals(snapshot.b2, model.b2)
    finally trainer.close()

  test("executionInfo reports an honest CUDA backend with device metadata"):
    assumeGpu()
    val trainer = CudaMlpTrainer(MlpConfig(inputSize = 3))
    try
      val info = trainer.executionInfo
      assertEquals(info.backend, MlBackend.Cuda)
      assert(info.deviceName.nonEmpty)
      assertEquals(info.precision, "float64")
    finally trainer.close()

  private def synthetic(rows: Int, seed: Long): (DenseMatrix, Array[Double]) =
    val random = java.util.Random(seed)
    val columns = 3
    val values = Array.fill(rows * columns)(random.nextDouble() * 2.0 - 1.0)
    val x = DenseMatrix(rows, columns, values)
    val y = Array.tabulate(rows) { row =>
      val a = values(row * columns)
      val b = values(row * columns + 1)
      val c = values(row * columns + 2)
      a * b + 0.3 * c * c - 0.2
    }
    (x, y)

  test("one GPU epoch matches the CPU reference update parameter by parameter"):
    assumeGpu()
    for l2 <- Vector(0.0, 0.001) do
      val (x, y) = synthetic(rows = 48, seed = 61L)
      val model = MlpModel.initialized(MlpConfig(inputSize = 3, hiddenSize = 8, seed = 8L))
      val adam = AdamConfig(learningRate = 0.01, l2 = l2)
      val gpu = CudaMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8), adam)
      val cpu = CpuMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8), adam)
      try
        gpu.loadModel(model)
        cpu.loadModel(model)
        val gpuMetrics = gpu.trainEpoch(gpu.upload(x, y))
        val cpuMetrics = cpu.trainEpoch(cpu.upload(x, y))
        assertEqualsDouble(gpuMetrics.mse, cpuMetrics.mse, tolerance, s"mse l2=$l2")
        assertEqualsDouble(gpuMetrics.objective, cpuMetrics.objective, tolerance, s"objective l2=$l2")
        val gpuModel = gpu.snapshotModel()
        val cpuModel = cpu.snapshotModel()
        assertClose(gpuModel.w1, cpuModel.w1, s"w1 l2=$l2")
        assertClose(gpuModel.b1, cpuModel.b1, s"b1 l2=$l2")
        assertClose(gpuModel.w2, cpuModel.w2, s"w2 l2=$l2")
        assertEqualsDouble(gpuModel.b2, cpuModel.b2, tolerance, s"b2 l2=$l2")
      finally
        cpu.close()
        gpu.close()

  test("five GPU epochs stay in lockstep with the CPU reference"):
    assumeGpu()
    val (x, y) = synthetic(rows = 64, seed = 71L)
    val adam = AdamConfig(learningRate = 0.02, l2 = 0.001)
    val gpu = CudaMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8, seed = 9L), adam)
    val cpu = CpuMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8, seed = 9L), adam)
    try
      val gpuBatch = gpu.upload(x, y)
      val cpuBatch = cpu.upload(x, y)
      for epoch <- 1 to 5 do
        val gpuMetrics = gpu.trainEpoch(gpuBatch)
        val cpuMetrics = cpu.trainEpoch(cpuBatch)
        assertEqualsDouble(gpuMetrics.mse, cpuMetrics.mse, 1e-8, s"epoch $epoch")
      val gpuModel = gpu.snapshotModel()
      val cpuModel = cpu.snapshotModel()
      for index <- gpuModel.w1.indices do
        assertEqualsDouble(gpuModel.w1(index), cpuModel.w1(index), 1e-7, s"w1($index)")
    finally
      cpu.close()
      gpu.close()

  test("the first update with L2 changes weights but leaves bias updates identical"):
    assumeGpu()
    val (x, y) = synthetic(rows = 32, seed = 81L)
    val model = MlpModel.initialized(MlpConfig(inputSize = 3, hiddenSize = 8, seed = 10L))
    val plain = CudaMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8), AdamConfig(l2 = 0.0))
    val regularized = CudaMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8), AdamConfig(l2 = 0.5))
    try
      plain.loadModel(model)
      regularized.loadModel(model)
      plain.trainEpoch(plain.upload(x, y))
      regularized.trainEpoch(regularized.upload(x, y))
      val plainModel = plain.snapshotModel()
      val regularizedModel = regularized.snapshotModel()
      assertClose(regularizedModel.b1, plainModel.b1, "b1")
      assertEqualsDouble(regularizedModel.b2, plainModel.b2, tolerance, "b2")
      val largestWeightShift = plainModel.w1.indices.map(index =>
        math.abs(plainModel.w1(index) - regularizedModel.w1(index))).max
      assert(largestWeightShift > 1e-9, s"weights unaffected by L2: $largestWeightShift")
    finally
      regularized.close()
      plain.close()

  test("GPU training reduces synthetic MSE and the trained model serves CPU inference"):
    assumeGpu()
    val (trainX, trainY) = synthetic(rows = 256, seed = 21L)
    val (evalX, evalY) = synthetic(rows = 128, seed = 22L)
    val trainer = CudaMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8, seed = 42L), AdamConfig(learningRate = 0.01))
    try
      val batch = trainer.upload(trainX, trainY)
      val first = trainer.trainEpoch(batch)
      var last = first
      for _ <- 2 to 300 do last = trainer.trainEpoch(batch)
      assert(last.mse < 0.2 * first.mse, s"train MSE only moved from ${first.mse} to ${last.mse}")
      val model = trainer.snapshotModel()
      val hostMse = {
        val predictions = CpuInference.predict(model, trainX)
        predictions.indices.map(i => (predictions(i) - trainY(i)) * (predictions(i) - trainY(i))).sum / predictions.length
      }
      assertEqualsDouble(last.mse, hostMse, 1e-8, "metrics are measured after the update")
      val evalPredictions = CpuInference.predict(model, evalX)
      val evalMse = evalPredictions.indices.map(i => (evalPredictions(i) - evalY(i)) * (evalPredictions(i) - evalY(i))).sum / evalPredictions.length
      val baseline = evalY.map(value => value * value).sum / evalY.length
      assert(evalMse < 0.3 * baseline, s"evaluation MSE $evalMse against baseline $baseline")
    finally trainer.close()

  test("loading a model on the GPU resets Adam moments and the step count"):
    assumeGpu()
    val (x, y) = synthetic(rows = 32, seed = 91L)
    val trainer = CudaMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8, seed = 11L))
    val fresh = CudaMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8, seed = 11L))
    try
      val batch = trainer.upload(x, y)
      trainer.trainEpoch(batch)
      trainer.trainEpoch(batch)
      val saved = trainer.snapshotModel()
      val continued = trainer.trainEpoch(batch)
      trainer.loadModel(saved)
      val reloaded = trainer.trainEpoch(batch)
      val freshBatch = fresh.upload(x, y)
      fresh.loadModel(saved)
      val reference = fresh.trainEpoch(freshBatch)
      assertEqualsDouble(reloaded.mse, reference.mse, 1e-12, "reset moments")
      assertNotEquals(reloaded.mse, continued.mse)
    finally
      fresh.close()
      trainer.close()

  test("lifecycle rules hold on the CUDA backend"):
    assumeGpu()
    val x = randomMatrix(2, 3, 2L)
    val y = Array(0.5, -0.5)
    val first = CudaMlpTrainer(MlpConfig(inputSize = 3))
    val second = CudaMlpTrainer(MlpConfig(inputSize = 3))
    try
      val batch = first.upload(x, y)
      assert(intercept[IllegalArgumentException](second.trainEpoch(batch)).getMessage.contains("different trainer"))
      assert(intercept[IllegalArgumentException](
        first.upload(DenseMatrix(2, 2, Array(1.0, 2.0, 3.0, 4.0)), y)).getMessage.contains("expects 3"))
      batch.close()
      batch.close()
    finally
      second.close()
      first.close()
    first.close()
    assert(intercept[IllegalArgumentException](first.predict(x)).getMessage.contains("closed"))
    assert(intercept[IllegalArgumentException](first.upload(x, y)).getMessage.contains("closed"))
    assert(intercept[IllegalArgumentException](first.snapshotModel()).getMessage.contains("closed"))

  test("trainers create and release GPU resources repeatedly"):
    assumeGpu()
    for round <- 1 to 3 do
      val trainer = CudaMlpTrainer(MlpConfig(inputSize = 4, hiddenSize = 4, seed = round.toLong))
      try
        val x = randomMatrix(16, 4, round.toLong)
        assertClose(trainer.predict(x), CpuInference.predict(trainer.snapshotModel(), x), s"round $round")
      finally trainer.close()
