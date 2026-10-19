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

  test("GPU training is reported as not yet implemented instead of silently using the CPU"):
    assumeGpu()
    val trainer = CudaMlpTrainer(MlpConfig(inputSize = 3))
    try
      val batch = trainer.upload(randomMatrix(4, 3, 1L), Array(1.0, 2.0, 3.0, 4.0))
      assert(intercept[UnsupportedOperationException](trainer.trainEpoch(batch)).getMessage.contains("not implemented"))
    finally trainer.close()

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
