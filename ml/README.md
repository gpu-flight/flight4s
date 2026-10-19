# flight4s-ml

A small neural-network regression module: a single-hidden-layer MLP
(`X -> Dense -> tanh -> Dense -> linear scalar`) with full-batch Adam training,
built for consumers that prepare their own numerical features. The consuming
application owns feature engineering, normalization, chronological splits and
model selection; this module owns parameters, forward propagation, loss,
backpropagation, Adam, persistence and inference.

Status: both backends are fully implemented. `CudaMlpTrainer(config, optimizer)`
runs forward propagation, backpropagation, gradient aggregation and Adam
updates on the GPU over one flat device-resident parameter buffer; the dataset
and parameters stay on the device across epochs, and only the per-epoch
predictions (plus one scalar when L2 is active) are downloaded for metrics.
GPU results are verified against the CPU reference within an absolute 1e-10 per
update (CUDA and JVM `tanh` may differ in the last ulps; Adam arithmetic is
otherwise written in the identical order), 1e-8 per epoch over multi-epoch
runs. Creation fails with an actionable error when CUDA or the native library
is unavailable; nothing silently falls back to the CPU. GPU tests are gated by
`-Dflight4s.cuda.native.path` and report skips otherwise.

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

## Integration

flight4s publishes no remote artifacts; consume the module as a pinned source
dependency. Either reference the project directly:

```scala
// In the consuming build.sbt, with flight4s checked out at a pinned commit:
lazy val flight4sMl = ProjectRef(file("../flight4s"), "ml")
lazy val app = (project in file(".")).dependsOn(flight4sMl)
```

or publish locally from the flight4s checkout and depend on the coordinates:

```bash
sbt ml/publishLocal   # publishes io.github.gpu-flight %% flight4s-ml % 0.1.0-SNAPSHOT
```

Supported and tested environment: Scala 3.8.1, sbt 1.12.2, JDK 25 (Amazon
Corretto), CUDA 13.3 on Windows x64 with an sm_120 device. The GPU backend
needs the flight4s native library: build it per the repository instructions
(CMake under `native/`, producing `flight4s_cuda.dll` or the platform
equivalent) and pass `-Dflight4s.cuda.native.path=<path to the library>` to the
JVM. CPU inference (`CpuInference`, `MlpModel.load`) needs neither CUDA nor the
native library.

Commands:

```bash
sbt ml/test                                                   # portable tests, no GPU needed
sbt -Dflight4s.cuda.native.path=<lib> ml/test                 # portable plus GPU tests
sbt -Dflight4s.cuda.native.path=<lib> "examples/runMain flight4s.examples.MlpRegressionExample 200"
```

## Known limitations

One hidden layer, scalar output, full-batch training, Double precision only.
No resumable optimizer checkpoints, no gradient clipping, no mini-batches.
