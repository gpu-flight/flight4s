package flight4s.examples

import java.nio.file.{Files, Path}
import flight4s.ml.*

/** End-to-end flight4s-ml walkthrough: synthetic data, GPU training, persistence, CPU inference.
  *
  * Run with the native library configured, optionally passing the epoch count and model path:
  *   sbt -Dflight4s.cuda.native.path=<flight4s_cuda library> "examples/runMain flight4s.examples.MlpRegressionExample 200"
  */
object MlpRegressionExample:
  private def synthetic(rows: Int, columns: Int, seed: Long): (DenseMatrix, Array[Double]) =
    val random = java.util.Random(seed)
    val values = Array.fill(rows * columns)(random.nextDouble() * 2.0 - 1.0)
    val x = DenseMatrix(rows, columns, values)
    val y = Array.tabulate(rows) { row =>
      val a = values(row * columns)
      val b = values(row * columns + 1)
      val c = values(row * columns + 2)
      a * b + 0.3 * c * c - 0.2
    }
    (x, y)

  private def mse(predictions: Array[Double], targets: Array[Double]): Double =
    predictions.indices.map(i => (predictions(i) - targets(i)) * (predictions(i) - targets(i))).sum / predictions.length

  def main(arguments: Array[String]): Unit =
    val epochs = arguments.headOption.map(_.toInt).getOrElse(200)
    val modelPath = arguments.lift(1).map(Path.of(_))
      .getOrElse(Files.createTempDirectory("flight4s-ml-example-").resolve("model.json"))
    val (trainX, trainY) = synthetic(rows = 512, columns = 14, seed = 42L)
    val (evalX, evalY) = synthetic(rows = 128, columns = 14, seed = 43L)

    // 1. Create an explicitly CUDA-backed trainer; this fails loudly when CUDA is unavailable.
    val trainer = CudaMlpTrainer(MlpConfig(inputSize = 14, hiddenSize = 8, seed = 7L),
      AdamConfig(learningRate = 0.01, l2 = 0.001))
    try
      println(s"backend: ${trainer.executionInfo}")

      // 2. Upload the full batch once and train epoch by epoch.
      val batch = trainer.upload(trainX, trainY)
      var last: TrainingMetrics = null
      for epoch <- 1 to epochs do
        last = trainer.trainEpoch(batch)
        if epoch == 1 || epoch % 50 == 0 then
          println(f"epoch ${last.epoch}%4d  mse ${last.mse}%.6f  objective ${last.objective}%.6f")

      // 3. Save the trained model to its portable JSON format.
      val model = trainer.snapshotModel()
      MlpModel.save(model, modelPath)
      println(s"saved model to $modelPath")

      // 4. Reload it and run CPU inference without touching CUDA.
      val reloaded = MlpModel.load(modelPath)
      val evalPredictions = CpuInference.predict(reloaded, evalX)
      println(f"evaluation mse via CPU inference: ${mse(evalPredictions, evalY)}%.6f")
      val gpuEval = trainer.predict(evalX)
      println(f"evaluation mse via GPU predict:   ${mse(gpuEval, evalY)}%.6f")
    finally
      // 5. Release every GPU resource deterministically.
      trainer.close()
      println("closed the trainer and released GPU resources")
