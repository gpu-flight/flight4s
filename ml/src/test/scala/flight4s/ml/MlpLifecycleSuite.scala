package flight4s.ml

import munit.FunSuite

class MlpLifecycleSuite extends FunSuite:
  private val x = DenseMatrix(2, 3, Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
  private val y = Array(1.0, -1.0)

  test("matrix and target validation produces clear messages"):
    val trainer = CpuMlpTrainer(MlpConfig(inputSize = 3))
    try
      assert(intercept[IllegalArgumentException](
        trainer.upload(DenseMatrix(0, 3, Array.empty), Array.empty)).getMessage.contains("positive row count"))
      assert(intercept[IllegalArgumentException](
        trainer.upload(DenseMatrix(2, 3, Array(1.0)), y)).getMessage.contains("does not match"))
      assert(intercept[IllegalArgumentException](
        trainer.upload(DenseMatrix(2, 2, Array(1.0, 2.0, 3.0, 4.0)), y)).getMessage.contains("expects 3"))
      assert(intercept[IllegalArgumentException](
        trainer.upload(DenseMatrix(2, 3, Array(1.0, 2.0, Double.NaN, 4.0, 5.0, 6.0)), y)).getMessage.contains("nonfinite"))
      assert(intercept[IllegalArgumentException](trainer.upload(x, Array(1.0))).getMessage.contains("y length"))
      assert(intercept[IllegalArgumentException](
        trainer.upload(x, Array(1.0, Double.PositiveInfinity))).getMessage.contains("nonfinite"))
    finally trainer.close()

  test("configuration validation rejects invalid settings and unsafe sizes"):
    assert(intercept[IllegalArgumentException](CpuMlpTrainer(MlpConfig(inputSize = 0))).getMessage.contains("inputSize"))
    assert(intercept[IllegalArgumentException](
      CpuMlpTrainer(MlpConfig(inputSize = 3, hiddenSize = 0))).getMessage.contains("hiddenSize"))
    assert(intercept[IllegalArgumentException](
      CpuMlpTrainer(MlpConfig(inputSize = Int.MaxValue, hiddenSize = 4))).getMessage.contains("exceeds"))
    assert(intercept[IllegalArgumentException](
      CpuMlpTrainer(MlpConfig(inputSize = 3), AdamConfig(learningRate = 0.0))).getMessage.contains("learningRate"))
    assert(intercept[IllegalArgumentException](
      CpuMlpTrainer(MlpConfig(inputSize = 3), AdamConfig(beta1 = 1.0))).getMessage.contains("beta1"))
    assert(intercept[IllegalArgumentException](
      CpuMlpTrainer(MlpConfig(inputSize = 3), AdamConfig(epsilon = 0.0))).getMessage.contains("epsilon"))
    assert(intercept[IllegalArgumentException](
      CpuMlpTrainer(MlpConfig(inputSize = 3), AdamConfig(l2 = -0.1))).getMessage.contains("l2"))

  test("close is idempotent and closed trainers reject every operation"):
    val trainer = CpuMlpTrainer(MlpConfig(inputSize = 3))
    val batch = trainer.upload(x, y)
    trainer.close()
    trainer.close()
    assert(intercept[IllegalArgumentException](trainer.upload(x, y)).getMessage.contains("closed"))
    assert(intercept[IllegalArgumentException](trainer.trainEpoch(batch)).getMessage.contains("closed"))
    assert(intercept[IllegalArgumentException](trainer.predict(x)).getMessage.contains("closed"))
    assert(intercept[IllegalArgumentException](trainer.snapshotModel()).getMessage.contains("closed"))
    assert(intercept[IllegalArgumentException](
      trainer.loadModel(MlpModel.initialized(MlpConfig(inputSize = 3)))).getMessage.contains("closed"))

  test("batches are rejected across trainers and after being closed"):
    val first = CpuMlpTrainer(MlpConfig(inputSize = 3))
    val second = CpuMlpTrainer(MlpConfig(inputSize = 3))
    try
      val batch = first.upload(x, y)
      assert(intercept[IllegalArgumentException](second.trainEpoch(batch)).getMessage.contains("different trainer"))
      batch.close()
      assert(intercept[IllegalArgumentException](first.trainEpoch(batch)).getMessage.contains("batch is closed"))
    finally
      second.close()
      first.close()

  test("shape edge cases train: single row, single feature, and the consumer widths"):
    for (rows, columns) <- Vector((1, 1), (1, 5), (3, 1), (513, 14), (64, 18), (65, 21)) do
      val random = java.util.Random(rows * 31L + columns)
      val features = DenseMatrix(rows, columns, Array.fill(rows * columns)(random.nextDouble() * 2.0 - 1.0))
      val targets = Array.fill(rows)(random.nextDouble() * 2.0 - 1.0)
      val trainer = CpuMlpTrainer(MlpConfig(inputSize = columns, hiddenSize = 8, seed = 1L))
      try
        val batch = trainer.upload(features, targets)
        val metrics = trainer.trainEpoch(batch)
        assert(java.lang.Double.isFinite(metrics.mse), s"$rows x $columns")
        assertEquals(trainer.predict(features).length, rows)
      finally trainer.close()
