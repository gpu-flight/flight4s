# flight4s-ml

A small neural-network regression module: a single-hidden-layer MLP
(`X -> Dense -> tanh -> Dense -> linear scalar`) with full-batch Adam training,
built for consumers that prepare their own numerical features. The consuming
application owns feature engineering, normalization, chronological splits and
model selection; this module owns parameters, forward propagation, loss,
backpropagation, Adam, persistence and inference.

Status: the CPU reference backend, model format and public API are implemented
(this slice). The CUDA trainer backend arrives in the next slices and will be
verified against this reference.

## Data contract

- `DenseMatrix(rows, columns, values)` is row-major: `values(row * columns + column)`.
- Training takes `x: DenseMatrix` (N x D) and `y: Array[Double]` (N targets);
  predictions return N values in row order.
- The caller keeps ownership of its arrays: `upload` copies them, and
  `snapshotModel()` returns copies that never alias live trainer state.
- Inputs, targets, parameters and configurations are validated eagerly with
  clear `IllegalArgumentException` messages: positive dimensions, matching
  lengths, finite values and overflow-checked sizes.
- No implicit normalization is performed, ever.

## Loss and optimizer

- `dataMse = (1 / N) * sum((prediction - target)^2)`.
- Optional coupled L2 through the gradient:
  `objective = dataMse + (l2 / 2) * (sum W1^2 + sum W2^2)`, biases excluded.
  This is classic L2, not decoupled AdamW weight decay.
- Adam with bias correction for both moments; defaults: learningRate 0.01,
  beta1 0.9, beta2 0.999, epsilon 1e-8, l2 0.0. One epoch = one full-batch
  update; the step count increments once per update. No gradient clipping.
- `TrainingMetrics(epoch, mse, objective)` is measured on a fresh forward pass
  after the epoch's update.
- `loadModel` installs parameters and resets Adam moments and the step count;
  it is a model snapshot, not a resumable training checkpoint.

## Numerical conventions

Double precision throughout. Accumulation order is defined and deterministic on
the CPU reference: each pre-activation starts from its bias and adds terms in
feature-index order; the output accumulates hidden terms in hidden-index order.
Initialization is Xavier uniform over `java.util.Random(seed)` (W1 then W2 in
index order, biases zero), reproducible across platforms because
`java.util.Random` is algorithmically specified.

## Model format

Versioned JSON written and parsed by a strict schema-specific codec (no
dependencies, no Java serialization). Fields: `format` ("flight4s-ml-mlp"),
`version` (1), `inputSize`, `hiddenSize`, `activation` ("tanh"), `output`
("linear"), `precision` ("float64"), `w1` (row-major D x H), `b1`, `w2`, `b2`.
Doubles are written with `Double.toString`, whose shortest decimal form
round-trips exactly, so saved models reproduce predictions bit-for-bit on the
same platform. Loading validates format, version, dimensions and finiteness.
`CpuInference.predict(model, x)` serves a loaded model without initializing
CUDA or any native library.

## Usage

```scala
import flight4s.ml.*

val config = MlpConfig(inputSize = 14, hiddenSize = 8, seed = 42L)
val trainer = CpuMlpTrainer(config, AdamConfig(learningRate = 0.01, l2 = 0.001))
try
  val batch = trainer.upload(x, y)
  for epoch <- 1 to 200 do
    val metrics = trainer.trainEpoch(batch)
    // Inspect validation performance externally and keep the best snapshot.
  val model = trainer.snapshotModel()
  MlpModel.save(model, path)
  val predictions = CpuInference.predict(MlpModel.load(path), features)
finally trainer.close()
```

Tests: `sbt ml/test` runs fully portable with no GPU. GPU-backed tests will use
the repository's `flight4s.cuda.native.path` gating when the CUDA trainer lands.

## Known limitations

One hidden layer, scalar output, full-batch training, Double precision only.
No CUDA backend yet (next slices), no resumable optimizer checkpoints, no
gradient clipping, no mini-batches.
