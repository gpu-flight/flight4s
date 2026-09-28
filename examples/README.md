# Flight4s Examples

These programs are runnable correctness examples, not production inference
operators or performance benchmarks. The examples module is not published.

## Row Softmax

[RowSoftmax.scala](src/main/scala/flight4s/examples/RowSoftmax.scala) contains the
complete DSL kernel and host `main`.

- One CUDA thread owns one row; columns are processed serially.
- Pass 1 computes the row maximum using a strict scalar fold.
- Pass 2 sums `exp(x - maximum)` and snapshots the denominator with `let`.
- Pass 3 stores normalized probabilities.
- The host compiles, loads, allocates, copies, launches on an owned stream,
  synchronizes, reads results, and closes the context with `Using.resource`.

`RowSoftmax.run(values, rows, columns, deviceOrdinal = 0)` accepts finite
row-major Float logits. Rows must be non-negative, columns positive, and the
shape must exactly match the array length. Empty batches return an empty array
without native setup. NaNs and infinities are rejected; special-value policies
and masked softmax are outside this example. Buffers are separate and this
entry point does not mutate the input array.

Each call creates a context wrapper, compiles, allocates, and performs synchronous
copies around an explicit-stream kernel launch. It intentionally exposes the
full lifecycle; it does not demonstrate reusable operator plans or overlap.
The core runtime still supports asynchronous launches and pinned transfers.

## Block-Cooperative Row Softmax

[BlockRowSoftmax.scala](src/main/scala/flight4s/examples/BlockRowSoftmax.scala)
keeps the serial example above intact and assigns one 128-thread block to each
row. `gpuRange(...).by(128)` gives each thread its strided columns. Scala
expression functions, folds, and sums build the per-thread work; explicit
shared-memory stores and block barriers combine those partial results.

- A Float scratch array uses 512 bytes per block and is reused between phases.
- Threads without columns contribute negative infinity to max and zero to sum.
- Two seven-stage trees combine the block maximum and denominator.
- Every barrier is outside lane-varying branches. An extra barrier protects
  scratch reuse after all threads snapshot the maximum.
- The fixed tree changes addition order from the serial example. Numerical
  tolerance, not bitwise equality with serial softmax, is the contract.

`BlockRowSoftmax.run(values, rows, columns, deviceOrdinal = 0)` uses the same
finite-input and shape validation and preserves the input array. It launches
exactly `(128, 1, 1)` threads per block. Its definition declares
`.requiringBlock(LaunchBlock.x(128))`, so typed launches of the exposed definition
also reject any other shape before native submission. This is a correctness example, not a benchmark
or a tuned replacement for cuDNN.

```shell
sbt "examples/runMain flight4s.examples.BlockRowSoftmax --cuda-source"
sbt -Dflight4s.cuda.native.path=<absolute-library-path> "examples/runMain flight4s.examples.BlockRowSoftmax"
```

The native test covers 13 fixtures and 74,191 probabilities per pass, repeats
every fixture, and compares against Double reference results (absolute error
at most 2e-6; row-sum error at most 2e-5). Widths range from 1 to 4,097, including
warp/block boundaries, tails, large offsets, equal logits, and Float extrema.
Racecheck, synccheck, and memcheck all reported zero issues on the tested GPU.
These observations are not a proof for every input, toolkit, or device.

For the installed Windows tooling, run the GPU fixture under Compute Sanitizer
by wrapping the Java sbt launcher so child JVMs are tracked:

```powershell
compute-sanitizer --tool racecheck --target-processes all --error-exitcode 1 `
  java "-Dflight4s.cuda.native.path=$native" -jar <sbt-launch.jar> `
  "examples/testOnly flight4s.examples.BlockRowSoftmaxJniSuite"
```

Repeat with `--tool synccheck` and `--tool memcheck`. Use the actual launcher and
native-library paths on your machine. Portable contracts run with
`sbt "examples/testOnly flight4s.examples.BlockRowSoftmaxSuite"`.

## Run the Serial Example

From the repository root, inspect generated CUDA without configuring JNI:

```shell
sbt "examples/runMain flight4s.examples.RowSoftmax --cuda-source"
```

Build the native library using [native/README.md](../native/README.md), then set
its absolute path. For example, in PowerShell from this repository:

```powershell
$native = (Resolve-Path native/build/Release/flight4s_cuda.dll).Path
sbt "-Dflight4s.cuda.native.path=$native" "examples/runMain flight4s.examples.RowSoftmax"
```

The built-in fixture uses rows with large positive logits, large negative logits,
and equal logits. Expected output (rounded):

```text
row 0: 0.0320586 0.0871443 0.2368828 0.6439142 (sum=1.0)
row 1: 0.0320586 0.0871443 0.2368828 0.6439142 (sum=1.0)
row 2: 0.2500000 0.2500000 0.2500000 0.2500000 (sum=1.0)
```

## Verify

Portable host-contract and code-generation tests:

```shell
sbt "examples/testOnly flight4s.examples.RowSoftmaxSuite"
```

GPU execution plus the complete project suite:

```powershell
$native = (Resolve-Path native/build/Release/flight4s_cuda.dll).Path
sbt "-Dflight4s.cuda.native.path=$native" test "examples/runMain flight4s.examples.RowSoftmax"
```

The native example test uses six fixtures: singleton rows, equal logits, large
offsets, extreme finite Float values, and partial final blocks with 129/257 rows.
It compares against a Double CPU reference with absolute error <= 2e-6 and
checks each row sum within 2e-5. This is not an exhaustive accuracy bound.
A skipped JNI test does not verify GPU execution.

Examples run and test tasks fork JVMs, forwarding `flight4s.cuda.native.path`
when set. This avoids loading the same JNI library in sbt's separate test/run
classloaders. See [sbt forking](https://www.scala-sbt.org/1.x/docs/Forking.html).
On newer JDKs, native-access warnings can appear on stderr even when a run exits
successfully; test summaries and the process exit status remain important.
