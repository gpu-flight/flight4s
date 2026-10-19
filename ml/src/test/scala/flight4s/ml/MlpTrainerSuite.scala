package flight4s.ml

import munit.FunSuite

class MlpTrainerSuite extends FunSuite:
  /** Fixed-seed nonlinear regression with an interaction term, per the acceptance criteria. */
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

  test("training substantially reduces MSE on synthetic data and generalizes to an evaluation set"):
    val (trainX, trainY) = synthetic(rows = 256, seed = 21L)
    val (evalX, evalY) = synthetic(rows = 128, seed = 22L)
    val trainer = CpuMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 8, seed = 42L), AdamConfig(learningRate = 0.01))
    try
      val batch = trainer.upload(trainX, trainY)
      val first = trainer.trainEpoch(batch)
      var last = first
      for _ <- 2 to 600 do last = trainer.trainEpoch(batch)
      assert(last.mse < 0.2 * first.mse, s"train MSE only moved from ${first.mse} to ${last.mse}")
      val evalPredictions = trainer.predict(evalX)
      val evalMse = MlpMath.meanSquaredError(evalPredictions, evalY)
      val evalBaseline = MlpMath.meanSquaredError(Array.fill(evalY.length)(0.0), evalY)
      assert(evalMse < 0.3 * evalBaseline, s"evaluation MSE $evalMse against baseline $evalBaseline")
      assertEquals(last.epoch, 600)
    finally trainer.close()

  test("predictions equal CPU inference from the snapshot and snapshots stay isolated"):
    val (x, y) = synthetic(rows = 32, seed = 31L)
    val trainer = CpuMlpTrainer(MlpConfig(inputSize = 3, seed = 5L))
    try
      val batch = trainer.upload(x, y)
      trainer.trainEpoch(batch)
      val snapshot = trainer.snapshotModel()
      val direct = trainer.predict(x)
      assertEquals(CpuInference.predict(snapshot, x).toVector, direct.toVector)
      snapshot.w1(0) += 100.0
      snapshot.b1(0) += 100.0
      assertEquals(trainer.predict(x).toVector, direct.toVector)
    finally trainer.close()

  test("uploads copy caller arrays so later mutation cannot change training"):
    val (x, y) = synthetic(rows = 16, seed = 41L)
    val pristine = CpuMlpTrainer(MlpConfig(inputSize = 3, seed = 9L))
    val mutated = CpuMlpTrainer(MlpConfig(inputSize = 3, seed = 9L))
    try
      val pristineBatch = pristine.upload(x, y)
      val copy = DenseMatrix(x.rows, x.columns, x.values.clone())
      val mutatedBatch = mutated.upload(copy, y.clone())
      copy.values(0) = 999.0
      assertEquals(mutated.trainEpoch(mutatedBatch).mse, pristine.trainEpoch(pristineBatch).mse)
    finally
      mutated.close()
      pristine.close()

  test("loading a model resets Adam moments and the step count as documented"):
    val (x, y) = synthetic(rows = 32, seed = 51L)
    val trainer = CpuMlpTrainer(MlpConfig(inputSize = 3, seed = 3L))
    val fresh = CpuMlpTrainer(MlpConfig(inputSize = 3, seed = 3L))
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
      assertEquals(reloaded.mse, reference.mse)
      assertNotEquals(reloaded.mse, continued.mse)
    finally
      fresh.close()
      trainer.close()

  test("loading rejects an incompatible architecture"):
    val trainer = CpuMlpTrainer(MlpConfig(inputSize = 3))
    try
      val wrong = MlpModel.initialized(MlpConfig(inputSize = 4))
      val error = intercept[IllegalArgumentException](trainer.loadModel(wrong))
      assert(error.getMessage.contains("does not match trainer"))
    finally trainer.close()

  test("the CPU trainer reports a CPU backend honestly"):
    val trainer = CpuMlpTrainer(MlpConfig(inputSize = 3))
    try
      assertEquals(trainer.executionInfo.backend, MlBackend.Cpu)
      assertEquals(trainer.executionInfo.precision, "float64")
    finally trainer.close()
