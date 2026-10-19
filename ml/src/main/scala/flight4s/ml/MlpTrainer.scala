package flight4s.ml

/** One uploaded full training batch; owned by the trainer that created it. */
trait TrainingBatch extends AutoCloseable:
  def rows: Int
  def columns: Int

/** Synchronous full-batch trainer; one trainEpoch is one Adam update over the whole batch.
  * loadModel installs parameters and resets Adam moments and the step count; it is not a
  * resumable training checkpoint. */
trait MlpTrainer extends AutoCloseable:
  def upload(x: DenseMatrix, y: Array[Double]): TrainingBatch
  def trainEpoch(batch: TrainingBatch): TrainingMetrics
  def predict(x: DenseMatrix): Array[Double]
  def snapshotModel(): MlpModel
  def loadModel(model: MlpModel): Unit
  def executionInfo: ExecutionInfo
