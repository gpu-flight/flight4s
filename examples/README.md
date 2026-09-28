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

## Run

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
