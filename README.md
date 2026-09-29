# Flight4s

Flight4s is a Scala 3 GPU programming platform built under the
[GPUFlight](https://github.com/gpu-flight) organization.

The first implementation is CUDA-first: its typed DSL and IR generate
inspectable CUDA C++ for compilation with NVRTC. The project name deliberately
does not bind the overall platform to one GPU vendor, leaving room for future
HIP or Metal backends after the CUDA architecture and backend boundary are
proven.

The goal is CUDA programming in a Scala style, including functional
composition: reusable typed expressions and staged transformations, with
explicit memory and synchronization effects. Raw CUDA is an interoperability
path alongside the Scala DSL.

## Scala-style CUDA

### Automatic Local Names

Kernel-local bindings no longer need a CUDA variable name. Scala identifiers
remain your handles; Flight4s chooses collision-free names in the generated CUDA:

```scala
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock}

val blockSums = kernel("blockSums",
  params(input[Float]("source"), output[Float]("sums"), value[Int]("count"))) { p =>
  val i = let { blockIdx.x * blockDim.x + threadIdx.x }
  val reduction = block.reduction[Float](LaunchBlock.x(256))
  val item = choose(i < p._3)(p._1(i).read)(literal(0.0f))
  val total = reduction.sum(item)
  when(threadIdx.x === literal(0)) { p._2(blockIdx.x) := total }
}
```

Launch this kernel with exactly `LaunchBlock.x(256)` and one output element
per block. Every thread participates in the reduction, including threads past
`count`, which contribute zero.

Unnamed overloads also cover `local(initial)`, `localArray[T](count)`,
`sharedArray[T](count)`, `sharedArray2D[T](rows, columns[, rowStride])`,
`sharedArray3D[T](depth, rows, columns[, rowStride])`,
`dynamicSharedArray[T]()`, `gpuFor(from, until[, step])`,
`gpuRange(from, until)`, and `reduction.reduceTree(value)(combine)`.
The bracketed arguments above mean optional overload arguments, not Scala syntax.
Existing explicit-name overloads remain available for debugging and inspection.
Automatic names are deterministic in generated CUDA, including across independent
kernel constructions, and avoid explicit names declared later or at module scope.

`let(expression)` and `let { expression }` both store one device snapshot.
In the explicit DSL, plain `val x = expression` binds a host expression tree.
Use `local` for mutable device state and `when`/`gpuIf` for device control flow.
The separate annotation prototype below adds snapshots for a limited subset
of Scala bindings.
Kernel/parameter/constant names remain explicit. The low-level `reduceSum`
primitive still takes an index name; use `gpuRange(from, until).map(f).sum(initial)`
for its unnamed expression form.

### Annotation Prototype

The optional `flight4s-frontend` module supports `@kernel` on a factory returning
the existing typed `Kernel[Args]`. Top-level `val` bindings of `Expr[T]` become
device snapshots automatically:

```scala
import scala.annotation.experimental
import flight4s.frontend.kernel
import flight4s.core.dsl.CudaDsl.{kernel as buildKernel, *}

@experimental
@kernel
def vectorAdd = buildKernel("vectorAdd", params(
    input[Float]("left"), input[Float]("right"),
    output[Float]("target"), value[Int]("count"))) { p =>
  val i = blockIdx.x * blockDim.x + threadIdx.x
  when(i < p._4) {
    p._3(i) := p._1(i).read + p._2(i).read
  }
}
```

The factory and callers require explicit `@experimental` opt-in; no global
`-experimental` flag is enabled. `input` is read-only and `output` is read-write.
Device control flow still uses the DSL. Nested snapshots use explicit `let`;
Scala `var`, `lazy val`, host/helper calls, external expression captures, and
implicit tuple/product snapshots are rejected by this prototype.
See the [frontend contract and tests](frontend/README.md).

### Automatic Result Names

Scalar, tuple, and case-class folds also support `foldLeft(initial)(step)`.
Use `orderedSum(initial)` for a stored, serial result from any scalar traversal:

```scala
// Inside a kernel body with count and a Float source buffer.
val positiveTotal = gpuRange(literal(0), count)
  .map(i => source(i).read)
  .filter(_ > literal(0.0f))
  .orderedSum(literal(0.0f))

val pair = gpuRange(literal(0), count)
  .foldLeft((literal(0), literal(1))) { (state, _) =>
    (state._2, state._1 + state._2)
  }
```

Tuple and case-class fields update simultaneously from the previous state.
The existing mapped `.sum(initial, policy)` remains expression-only;
`orderedSum(initial)` emits statements and cannot be used in expression-only
callbacks. Neither API implicitly parallelizes a traversal.

Warp reductions, shuffles, votes, and atomic snapshots also accept unnamed forms:

```scala
// Inside a convergent kernel body; item is Expr[Float], counter is an Int buffer.
import flight4s.core.ir.MemoryOrder
import flight4s.core.types.UInt

val warpTotal = warp.reduceSum(UInt.fromBits(-1), item)
val ticket = atomic.device(counter(literal(0)))
  .fetchAdd(literal(1), MemoryOrder.Relaxed)
```

Warp participation masks and optional widths are unchanged. Atomic memory orders
and scopes remain explicit, and compare-exchange returns the observed old value,
not a Boolean. Explicit-name overloads remain available for every operation.

### Native Vector Values

`Float2` and `Float4` are actual CUDA `float2`/`float4` values, each owned by
one thread. A typed buffer index selects one whole vector:

```scala
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.Float4

val scaleVectors = kernel("scaleVectors",
  params(input[Float4]("source"), output[Float4]("out"), value[Int]("count"))) { p =>
  val i = let("i", blockIdx.x * blockDim.x + threadIdx.x)
  when(i < p._3) {
    val v = let("v", p._1(i).read)
    val bias = float4(literal(1f), literal(2f), literal(3f), literal(4f))
    p._2(i) := v.map(_ * literal(2f)).zipWith(bias)(_ + _)
  }
}
```

Use `.x`/`.y` on either type, and `.z`/`.w` on `Float4`. `map` and `zipWith`
stage component-wise Float expressions; they do not imply SIMD arithmetic or
memory coalescing. Callbacks are expression-only and run once per component
during staging. Use `let` to load once; ordinary reused expressions can reload.

Host `Array[Float2]` and `Array[Float4]` copies pack components in x/y/z/w order
using 8- and 16-byte elements respectively, with matching CUDA alignments.
This copies JVM values into native storage, not JVM case-class object layouts.
CUDA compilation needs the toolkit include directory for `vector_types.h`,
supplied through `CompilerOptions.additionalNvrtcOptions` as `--include-path=...`.
By-value vector kernel parameters, unchecked scalar-buffer casts, automatic
vectorization, and packed low-precision arithmetic are not provided by this API.

### Functional Ranges

Compose typed expressions with ordinary Scala functions, then stage loops and
reductions into the validated CUDA IR:

```scala
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Expr

def square(x: Expr[Float]): Expr[Float] = x * x

val rowSquares = kernel(
  "rowSquares",
  params(input[Float]("source"), output[Float]("sums"),
    value[Int]("rows"), value[Int]("columns"))
) { bindings =>
  val (source, sums, rows, columns) = bindings
  val row = local("row", blockIdx.x * blockDim.x + threadIdx.x)
  when(row.read < rows) {
    sums(row.read) := gpuRange("column", literal(0), columns)
      .map(column => source(row.read * columns + column).read)
      .map(square)
      .sum(literal(0.0f))
  }
}
```

`gpuRange` is half-open and defaults to unit stride. Use `.by(4)` before mapping
or filtering to select a positive static stride. Each CUDA thread executes its range
serially; the example explicitly assigns one row per thread. `map` composes
expression builders without allocating an intermediate device collection.
`sum` lowers to `ReduceSum`, preserving its explicit initial value, accumulator
type, and reduction policy. `foreach` lowers to the existing GPU loop for
explicit stores and other statements.

Indexing intrinsics `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` expose `Expr[Int]`
on all three axes. Generated CUDA explicitly casts each built-in field to `int`,
so negative comparisons, division, and remainder keep signed semantics even
though CUDA's underlying fields are unsigned. For an eight-thread block,
`literal(-7) / blockDim.x` is `0` and its remainder is `-7`. This does not add
overflow checks or promise JVM wraparound for general arithmetic; keep index
calculations within the signed 32-bit range.

`gridDim` reads the current launch's grid dimensions. For a multidimensional
grid, construct a linear block index without passing duplicate shape arguments:

```scala
val linearBlock = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)
val blockCount = gridDim.x * gridDim.y * gridDim.z
```

These are staged device expressions, not host values. Grid dimensions are
uniform across a launch but remain dynamic between launches, even for kernels
with a required block shape. Using them does not automatically distribute a
functional range across threads or enable dynamic `.by` steps.

Explicit statement loops also accept a positive static stride:

```scala
gpuFor("column", literal(0), columns, step = 4) { column =>
  result(column) := source(column).read
}
```

This kernel-body snippet visits `0, 4, 8, ...` below `columns`. The step is a
host `Int`, not a device expression. Zero and negative steps fail validation.
Non-unit loops use a private wide counter to avoid overflow on the final
increment; the callback index remains `Expr[Int]`. The lower bound is evaluated
once and the upper bound is rechecked, so use `let` to explicitly snapshot a
mutable endpoint. The same stride works with functional composition:

```scala
val total = gpuRange("column", literal(0), columns).by(4)
  .map(column => source(column).read)
  .sum(literal(0.0f))
```

`.by` returns a new range; `.by(2).by(3)` selects 3, not 6. Mapped/filtered
traversals, scalar/pair folds, and nested `for` generators retain the stride.
Invalid steps are diagnosed at terminal validation, with the terminal's source
location. The step does not assign work to threads or imply a parallel reduction.

Scala callbacks run during IR construction, not on the device. Keep `map`
callbacks expression-only. DSL statements and shared declarations inside map,
reduction, fold-step, or `choose`-arm callbacks raise a source-located
`DslError(StatementInsideExpression)` instead of escaping into an outer block.
This staging check does not prohibit ordinary JVM mutation or I/O.
Reusing an `Expr` does not snapshot or memoize a
load. Use `let` for a named, read-only device snapshot and `local` for mutable
device state:

```scala
val previous = let("previous", data(index).read)
data(index) := literal(0.0f)
result(index) := previous // Reads the saved value, not the updated buffer.
```

This snippet belongs inside a kernel body with the named parameters declared.
`let` stages a local declaration at that position; it is not an expression-only
combinator and cannot be hidden inside a `map` callback or `choose` arm.
Automatic grid distribution and parallel reductions are not implied by this API.

Reusable statement helpers can use `scoped { ... }` to isolate temporary locals
between calls. Each body becomes a lexical CUDA block, with the same explicit
stores and synchronization rules. Outer bindings remain visible; active names
cannot be shadowed, and scope-local values cannot be used after the scope ends.

`choose(condition)(whenTrue)(whenFalse)` returns a typed expression, so GPU
conditionals can compose in functions and maps:

```scala
def positivePart(x: Expr[Float]): Expr[Float] =
  choose(x > literal(0.0f))(x)(literal(0.0f))
```

Both arms must have the same CUDA type. Both expression trees are constructed
and validated on the host, but only the selected value is evaluated on the GPU.
Use `gpuIf` or `when` for conditional statements and stores instead.

Staged Booleans support `&&`, `||`, and `!` with GPU short-circuit evaluation:

```scala
val safe = (divisor !== literal(0)) && literal(12) / divisor > literal(2)
when(safe || fallback) { /* GPU statements */ }
```

Both operands must be `Expr[Boolean]`; use `when`, not a host Scala `if`.
The right-hand builder still runs once on the JVM under the expression-staging
guard. Parenthesize `!==` comparisons in compound conditions because Scala
gives that operator assignment-level precedence. Use DSL `===` / `!==` for
device comparison, not host `==` / `!=`.

Signed `Int` and unsigned `UInt` expressions support Scala-style bitwise `&`,
`|`, and `^`. Both operands must have the same type; unlike Boolean `&&`/`||`,
these operators evaluate both operands. They compose in expression callbacks:

```scala
val maskedTotal = gpuRange("i", literal(0), literal(32))
  .map(i => i & literal(7))
  .filter(i => (i ^ literal(1)) > literal(0))
  .foldLeft("maskedTotal", literal(0))(_ + _)
```

Use parentheses around bitwise expressions when combining them with comparisons.
Unsigned `UInt` values also support `<<`, `>>`, and `>>>` with `Expr[Int]`
counts. Both unsigned right shifts zero-fill; counts use their low five bits,
so shifting by 32 is the same as shifting by 0. For example:

```scala
val oneBit = literal(UInt.fromBits(1)) << (threadIdx.x & literal(31))
```

Signed `Int` values support the same spellings and count masking: `<<` wraps
the 32-bit result, `>>` sign-extends, and `>>>` zero-fills while returning `Int`.
For example, `(word >>> bit) & literal(1)` extracts a bit and composes in
`map`/`filter`/`foldLeft`. Generated CUDA avoids undefined signed C++ shifts.
Unary `~` complements an `Int` or `UInt` word without changing its type:
`flags & (~disabled)` clears selected bits. It lowers through existing XOR IR,
so the generated CUDA may show an XOR with all ones.

Use `bits.popCount(word)` to count set bits in an Int or UInt expression.
It returns `Expr[Int]` in 0..32 and emits CUDA's `__popc` intrinsic. For
example, `bits.popCount(selectedMask)` counts the selected lanes in a ballot
result; the count itself is a per-thread expression, not a collective call.

Generated kernels currently compile as CUDA C++20. The explicit unsigned
intermediates used for signed shifts also avoid older C++17 shift pitfalls.

For a strict scalar recurrence, use a named `foldLeft` inside a kernel body:

```scala
val maximum = gpuRange("i", literal(0), count)
  .map(i => input(i).read)
  .foldLeft("maximum", literal(Float.NegativeInfinity)) { (best, x) =>
    choose(x > best)(x)(best)
  }
```

The fold stages an initialized local and a serial loop at its call site. Its
returned expression reads that result; repeated reads do not repeat the fold.
Each state component has a fixed CUDA scalar type, empty ranges return the initial value, and no
parallel reassociation is implied. Floating-point comparison behavior remains
explicit in the step function.

Pair state supports combined statistics and cross-dependent recurrences:

```scala
val (count, total) = gpuRange("i", literal(0), n)
  .map(i => input(i).read).filter(_ > literal(0.0f))
  .foldLeft("positive", (literal(0), literal(0.0f))) { (state, x) =>
    (state._1 + literal(1), state._2 + x)
  }
```

Both next components are evaluated before either state is updated. The pair
is a Scala tuple of scalar expressions, not a CUDA struct or a device collection.
The prefix above reserves `positive_0`, `positive_1`, `positive_next_0`, and
`positive_next_1` in their respective scopes. Existing name-conflict checks apply.
Nonempty tuple state extends the same rule beyond pairs:

```scala
val (count, total, squaredTotal) = gpuRange("i", literal(0), n)
  .map(i => input(i).read)
  .foldLeft("stats", (literal(0), literal(0.0f), literal(0.0f))) { (state, x) =>
    (state._1 + literal(1), state._2 + x, state._3 + x * x)
  }
```

Tuple folds work through plain, mapped, filtered, and nested traversals.
`Tuple1` and tuples beyond 22 components are supported (23 components are tested
on CUDA). Every component must remain an `Expr` of its original scalar type;
empty tuples, host-valued components, and changing tuple shapes are rejected.
All next values are snapshotted before stores, inside the accepted element's
guards. This is per-thread recurrence semantics, not atomicity between threads.
Large state can increase register/local-memory pressure. Parallel folds remain
separate work.

For named state fields, derive `ProductFoldState` on a nonempty case class:

```scala
import flight4s.core.dsl.ProductFoldState
case class Stats(count: Expr[Int], total: Expr[Double]) derives ProductFoldState

val stats = gpuRange("i", literal(0), n)
  .foldLeft("stats", Stats(literal(0), literal(0.0))) { (s, i) =>
    s.copy(count = s.count + literal(1), total = s.total + convert.i32ToF64(i))
  }
```

All fields must be scalar `Expr` values. Empty, nested, host-valued, and sum-type
state is rejected. Fields are read-only device expressions; all next fields are
snapshotted before any update. The case class exists only while staging on the
JVM: generated CUDA uses the same numbered scalar locals as tuple state, with no
struct, allocation, or ABI change. Keep constructors and callbacks pure; hidden
DSL statements in steps or reconstructed constructors are rejected. Ordinary
JVM effects are not prohibited. Generic classes can supply an explicit
`ProductFoldState.derived` instance with the necessary `CudaType` evidence.

Traversal elements can also be flat tuples of scalar expressions:

```scala
val indexed = gpuRange("i", literal(0), n)
  .map(i => (i, data(i).read))
  .filter { case (_, x) => x > literal(0.0f) }
indexed.foreach { case (i, x) => result(i) := x * literal(2.0f) }
```

Tuple `map`, guards, `flatMap`, `foreach`, and scalar/tuple/case-class folds
compose without intermediate device collections. Scala `for` generators can
yield tuples or destructure them in later generators. Every field must be a
scalar `Expr`; empty tuples, nested fields, and host values are rejected.
`Tuple1` and tuples above 22 elements are supported. Tuple fields still contain
expression trees, not snapshots: use `let` in a statement body when a load must
be preserved across stores. Mapping a tuple back to a scalar gives a general
serial traversal with `foldLeft`; expression-only `sum` remains limited to the
original unfiltered mapped range.

Traversal elements can use named case-class fields with the same derivation:

```scala
case class Sample(index: Expr[Int], value: Expr[Float]) derives ProductFoldState

val selected = gpuRange("i", literal(0), n)
  .map(i => Sample(i, data(i).read))
  .filter(_.value > literal(0.0f))
val total = selected.foldLeft("total", literal(0.0f))((sum, sample) => sum + sample.value)
selected.foreach(sample => result(sample.index) := sample.value * literal(2.0f))
```

Scalar, tuple, and product pipelines can map or expand into one another. Named
products support guarded `for` generators and scalar/tuple/product folds. They
remain flat, nonempty containers of `Expr` fields on the JVM, not CUDA structs
or stored snapshots. Each terminal emits a separate traversal. Nested product
fields, including comprehension value bindings that need nested transport,
remain unsupported; use an explicit flat `map` instead.

Ranges also support staged guards, including Scala `for` syntax:

```scala
for
  i <- gpuRange("i", literal(0), count)
  if input(i).read !== literal(0)
do output(i) := literal(12) / input(i).read
```

Inside a kernel, this generates a serial loop with a CUDA `if`. `filter` and
`withFilter` can be chained with maps, `foreach`, and strict `foldLeft`; later
guards and effects run only for accepted elements. No compacted device array
is allocated. An explicit named sum supports filtered and nested scalar pipelines:

```scala
val selectedTotal = gpuRange("i", literal(0), count)
  .filter(_ > literal(0))
  .map(i => input(i).read)
  .sum("selectedTotal", literal(0.0f))
```

Here `input` may contain `Float`, FP16, BF16, or either FP8 format; low-precision
elements are promoted to the `Float` accumulator before addition. Named sums
declare a local and add accepted elements in serial traversal order. Empty or
all-rejected traversals retain the initial value, without adding artificial
zeros. They require a statement context and do not take a `ReductionPolicy`.
The existing unfiltered `sum(initial, policy)` remains an expression-only
reduction with explicit policy metadata. Neither API implies a parallel tree.

Multiple generators use staged `flatMap` to express dependent nested loops:

```scala
val lowerTriangle = for
  i <- gpuRange("i", literal(0), n)
  j <- gpuRange("j", literal(0), i)
yield input(i * n + j).read

val total = lowerTriangle.foldLeft("total", literal(0.0f))(_ + _)
```

Each thread visits outer indices first, then inner indices in ascending order.
Maps, guards, further `flatMap`, `foreach`, and scalar/tuple folds can compose.
The inner factory builds once per terminal on the JVM; its bounds remain device
expressions. Scalar, tuple, and named-product elements are supported.
Comprehension value bindings must fit the supported flat expression shape;
expression-only flattened sums and automatic parallel distribution are not included.

Typed device math includes `exp`, `log`, `sqrt`, `rsqrt`, and `tanh` for `Float`
and `Double` expressions. Low-precision values require explicit promotion, for
example `exp(halfValue.toAccumulator[Float])`. These emit standard CUDA math
calls; no JVM constant folding or automatic fast-math option is introduced.

Conversion IR is validated before code generation: unsupported type pairs,
rounding modes, or saturation requests are rejected instead of being ignored.
F32-to-F16/BF16 supports all four rounding modes without finite saturation;
F32-to-FP8 supports nearest-even with either saturation policy. Exact widening
and identity conversions use nearest-even/no-saturation metadata. Existing
public conversion helpers select supported policies automatically.

Runtime integer sizes can be converted explicitly for floating-point formulas:

```scala
val mean = choose(count > literal(0))(total / convert.i32ToF32(count))(literal(0.0f))
```

`convert.i32ToF32` and `convert.u32ToF32` default to nearest-even and accept an
explicit rounding mode. `convert.i32ToF64` and `convert.u32ToF64` are exact for
all 32-bit inputs. These convert numeric values, not raw bits; no implicit
conversion between already-staged integer and floating expressions is added.

Use `convert.f32ToF64(x)` to widen a Float expression before Double arithmetic:

```scala
val wide = convert.f32ToF64(x)
val squared = wide * wide
```

This emits `static_cast<double>(x)` and preserves finite Float values exactly.
Widen before computing: converting `x * x` afterward cannot undo Float overflow
or rounding. NaN classification
is retained, but NaN payloads are not part of this API's contract.

To return Float output after wider accumulation, narrow explicitly:

```scala
val result = convert.f64ToF32(doubleResult) // nearest-even
val downward = convert.f64ToF32(doubleResult, RoundingMode.TowardNegative)
```

All four rounding modes are supported through CUDA `__double2float_*`
intrinsics. This can lose precision, underflow, or overflow; it does not clamp
finite values. No implicit narrowing or JVM evaluation is introduced.

## Atomic Updates

Use `atomicAdd` when several threads update the same global or shared element:

```scala
val histogram = kernel("histogram", params(inOut[Int]("counts"), value[Int]("n"))) { bindings =>
  val (counts, n) = bindings
  val i = let("i", blockIdx.x * blockDim.x + threadIdx.x)
  when(i < n) { atomicAdd(counts(i % literal(16)), literal(1)) }
}
```

Initialize the 16 counters before launch. `atomicAdd` supports `Int`, `UInt`, and
`Float` in read-write global buffers and shared arrays. It returns `Unit`, so it
can appear in a `foreach` body but not an expression-only callback. Generated
CUDA uses `::atomicAdd`; the previous value is discarded.

Atomicity does not initialize memory, synchronize a block, or publish unrelated
memory writes. Shared initialization and consumption still require appropriate
barriers. Floating-point results can depend on thread execution order.

Use a named capture when the old value is needed, for example to allocate a
unique output slot inside a kernel:

```scala
val ticket = atomicFetchAdd("ticket", counter(literal(0)), literal(1))
result(ticket) := input(index).read
```

The capture executes once and returns an `Expr` reading its local snapshot;
reusing `ticket` never repeats the update. The name follows normal lexical-scope
rules, and capture is still forbidden inside expression-only callbacks. Size the
output for all possible contributions, initialize the counter before launch,
and synchronize before consuming output. Ticket assignment is not stable input
order, and incrementing the counter does not publish the later output store.

### Scoped Atomic References

Use `atomic.device(place)` for a read-write global element, or
`atomic.block(place)` for read-write shared/global storage used within one block:

```scala
import flight4s.core.ir.MemoryOrder

val counterRef = atomic.device(counter(literal(0)))
val ticket = counterRef.fetchAdd("ticket", literal(1), MemoryOrder.Relaxed)
val current = counterRef.load("current", MemoryOrder.Acquire)
// After writing the associated payload, publish its ready flag:
atomic.device(ready(literal(0))).store(literal(1), MemoryOrder.Release)
```

These are kernel-body snippets with declared read-write buffers, not JVM atomics.
`Int`, `UInt`, `Float`, and `Double` support load/store, exchange, fetchAdd, and
fetchSub. Int/UInt also support fetchMin/Max/And/Or/Xor and compareExchange.
Each returned Expr is a named old-value snapshot; the operation runs once.
CompareExchange returns the observed old integer, not a Boolean, and takes
explicit success/failure orders. Compare it with a stable expected value to
determine success; it performs no retry loop.

Every operation requires an explicit order. Loads allow Relaxed, Acquire, or
SequentiallyConsistent; stores allow Relaxed, Release, or SequentiallyConsistent.
Read-modify-write operations also allow AcquireRelease. Invalid load/store
orders and unsupported compare-exchange order pairs fail IR validation.

This API requires CUDA 12.8+ and compute capability 6.0+, checked in emitted
CUDA without extra headers. The older atomicAdd/atomicFetchAdd API is unchanged.
Scopes must include all participating threads on both sides of a synchronization
relationship. Block scope on global storage is not cross-block synchronization.
Concurrent ordinary accesses to an atomic location remain unsafe without
separate synchronization. No system/cluster scope, fence API, progress guarantee,
automatic initialization, or implicit barrier is added.

## Warp Votes

Warp votes are named, once-evaluated collective statements:

```scala
import flight4s.core.types.UInt

val fullMask = literal(UInt.fromBits(-1))
val accepted = warp.ballot("accepted", fullMask, predicate)
val allAccepted = warp.all("allAccepted", fullMask, predicate)
val anyAccepted = warp.any("anyAccepted", fullMask, predicate)
```

Use these inside a kernel body. `predicate` is `Expr[Boolean]`; ballot returns
an `Expr[UInt]` snapshot, while all/any return Boolean snapshots. Every calling
lane must be included in its mask, and participating non-exited lanes must reach
the same call with a consistent mask. A full mask is not valid inside an
arbitrary per-thread branch. For a tail, vote before branching on the resulting
participation predicate or supply a correctly constructed subgroup mask.

Validation checks types, scope, and literal empty masks, not the full runtime
participation contract. Votes can compose in explicit statement loops but cannot
be hidden in `map` or `choose` callbacks. They are collective effects, not block
barriers or memory fences.

Shuffles capture another participating lane's scalar value:

```scala
val firstValue = warp.shuffle("firstValue", fullMask, value, literal(0), width = 32)
val previous = warp.shuffleUp("previous", fullMask, value, literal(UInt.fromBits(1)))
val next = warp.shuffleDown("next", fullMask, value, literal(UInt.fromBits(1)))
val partner = warp.shuffleXor("partner", fullMask, value, literal(1))
```

Supported values are Int, UInt, Float, and Double. Width is a host integer from
1, 2, 4, 8, 16, 32; the source lane is a nonnegative `Expr[Int]` relative to that
width-sized subgroup, with CUDA modulo-width wraparound. The selected lane must
actually participate in the mask. Width does not make inactive lanes available.

Up/down use an unsigned delta and keep the caller's value when crossing the
width-sized subgroup boundary. XOR uses a signed lane mask; it can read earlier
subgroups, but attempts to read later subgroups return the caller's value.
Directional deltas and lane masks must be in `0..31`; literal violations are
rejected, while dynamic validity remains the caller's responsibility.
Each shuffle result is a once-evaluated read-only snapshot, not a memory fence.
Explicit tree reductions are also available:

```scala
val total = warp.reduceSum("total", UInt.fromBits(-1), value, width = 32)
val product = warp.reduceTree("product", UInt.fromBits(-1), value, width = 8)(_ * _)
```

These support Int, UInt, Float, and Double. The mask is a **host UInt**, not a
dynamic expression, and must contain complete aligned width-sized groups.
The fixed adjacent-pair tree preserves left/right operand order in every lane;
it is not a sequential `foldLeft`. The combine must be statement-free and use
only its operands and lane-uniform captured values. All named non-exited lanes
must reach the same call with the same mask, and every calling lane must be
named. Static mask checks do not prove this runtime participation contract.
Width one captures the input without shuffling. Floating sums follow the tree's
rounding order; signed integer intermediates must not overflow. No memory fence,
implicit widening, sparse-mask reduction, or automatic parallelization is added.

Use the Unit-returning `warp.sync(mask)` to order shared/global memory accesses among participating
warp lanes. Unlike shuffles and votes, it is a memory-ordering synchronization
point. It does not synchronize another warp or the whole block. When reusing
scratch memory in a loop, synchronize after writing and again after all readers
have captured their values, before the next iteration overwrites the scratch.

Validation reports `WarpParticipationMayDiverge` when a shuffle, vote, or warp
barrier uses known lane-varying control flow or a lane-varying mask. The analysis
follows local declarations, assignments, branches, and loops. This is a warning,
not a proof of invalid participation: a correctly guarded partial-mask subgroup
can also warn. Conversely, no warning does not prove that every calling lane is
in the mask, that all named lanes reach the call, or that a shuffle source is
active. Unknown uniformity remains unproven. Generated CUDA is unchanged.

## Block Reductions

Declare reusable scratch at kernel scope, then call it collectively:

```scala
val reduction = block.reduction[Float]("scratch", LaunchBlock.x(128))
val total = reduction.sum("total", item)
val product = reduction.reduceTree("product", item)(_ * _)
```

Here `LaunchBlock` is `flight4s.core.launch.Block` imported with an alias, and
`item` is a per-thread expression. Scratch uses one static shared element per
thread. The block size must be a power of two from 1 through 1,024. Every thread
must call each reduction in the same order; pad missing inputs with the chosen
operation's identity instead of guarding only some threads around the call.

The kernel automatically requires the exact declared x/y/z shape. A conflicting
`.requiringBlock(...)` override fails validation, and typed launches reject a
different shape. Calls use an ordered adjacent-pair tree, capture the result in
every thread, and finish with a barrier before scratch may be reused. `sum`
requires an additive type; custom combines must be statement-free. No implicit
widening, tuple-state reduction, non-power-of-two tree, cross-block reduction,
or automatic conversion of `foldLeft` is implied.

## Status

Flight4s is pre-alpha and under active design. The current implementation
provides:

- CUDA scalar type witnesses, including F16, BF16, and FP8 formats;
- typed expressions, places, statements, control flow, and reductions;
- lazy staged `gpuRange.map` composition with typed `sum` and `foreach` terminals;
- lexical `scoped` bodies for reusable higher-order Scala statement helpers;
- typed value-producing conditionals with guarded CUDA expression evaluation;
- staged Boolean `&&`, `||`, and `!` with short-circuit device evaluation;
- strict named scalar/pair `foldLeft` terminals for ordered per-thread recurrences;
- named `let` snapshots returning read-only expressions for once-evaluated values;
- lazy filtered ranges and Scala `for` guards with ordered conditional execution;
- staged `flatMap` and multi-generator `for ... yield` for dependent nested loops;
- typed Float/Double device math with explicit low-precision promotion;
- typed Int/UInt/Float atomic addition in global and shared memory;
- named warp ballot/all/any snapshots with explicit participation masks;
- distinct module constants, rank-aware kernel shared arrays, and lexical local
  arrays;
- module and kernel validation for memory ownership, scope, access, and static
  literal bounds on constant, local, and shared arrays;
- typed kernel signatures and compile-time-checked launch argument tuples;
- ordered CUDA ABI descriptors and exact scalar/device-pointer byte encoding;
- aligned direct launch storage with stable descriptor codes and slot offsets;
- validated 1D, 2D, and 3D grid/block configuration with optional clusters;
- a JNI-facing launch request with reference metadata validation;
- a C++ JNI launcher with native ABI validation and `void**` construction;
- ordinary and clustered `cuLaunchKernelEx` execution without implicit sync;
- structural validation independent of code generation;
- validation-gated deterministic CUDA C++ source generation;
- inspectable generated module and typed kernel artifacts with explicit C++20
  compiler options;
- explicit compute-capability targets and ordered NVRTC option resolution;
- native NVRTC compilation with inspectable PTX, compiler logs, compiler
  version, target, and exact generated-source provenance;
- structured NVRTC compilation failures that retain source, options, and logs;
- public `CudaContext`, `CudaModule`, `CudaStream`, `CudaEvent`, and typed
  `CudaFunction[Args]` resources;
- retained CUDA primary contexts with compute-capability discovery;
- typed context-owned `CudaDeviceBuffer[T]` allocation and deterministic
  `AutoCloseable` cleanup;
- typed context-owned `CudaPinnedBuffer[T]` allocation backed by reusable
  page-locked host memory;
- exact whole-buffer and partial-range synchronous transfers between
  same-context pinned and device buffers;
- context-owned completion events with recording, non-blocking queries, host
  synchronization, and same-context stream dependencies;
- exact whole-buffer synchronous host/device copies through native-order direct
  staging storage;
- host codecs for every current scalar type, preserving raw F16, BF16, and FP8
  representations;
- context-owned default-mode and non-blocking CUDA streams with explicit
  synchronization;
- whole-buffer and partial-range asynchronous pinned-memory transfers on
  explicit streams;
- reverse-creation-order cleanup across context-owned modules, buffers, and
  streams;
- typed function resolution that preserves generated or raw kernel signatures and
  retains Driver-reported resource attributes;
- typed `CudaFunction.launch` submission from the original
  `KernelInvocation[Args]` or `RawCudaInvocation[Args]`;
- source-compatible default-stream launch and same-context explicit-stream
  launch with automatic in-flight resource retention;
- explicit context synchronization through `CudaContext.synchronize()`;
- launch-time kernel provenance, optional exact block-shape contracts, and
  dynamic shared-memory validation;
- context-scoped `cuLaunchKernelEx` with structured CUDA Driver failures;
- structured CUDA Driver failures with PTX JIT information and error logs;
- CUDA declaration emission for constants, global parameters, static and
  dynamic shared memory, and lexical local memory;
- a generated, compiled, and executed typed `vectorAdd` integration test;
- a handwritten raw CUDA `vectorAdd` compiled, loaded, and executed through the
  same typed runtime with CPU-reference verification;
- a generated, compiled, and executed dynamic shared-memory block reduction
  with CPU-reference verification;
- golden-source tests, native NVRTC contract tests, JNI integration tests, and
  an optional `nvcc` compilation test.

### Execution and resource lifetimes

Cooperative kernels can declare their required block geometry:

```scala
import flight4s.core.launch.{Block as LaunchBlock}
val definition = kernel("cooperative") { /* staged kernel body */ }
  .requiringBlock(LaunchBlock.x(128))
```

Typed launch rejects a different `(x, y, z)` with `BlockShapeMismatch` before
argument packing or native submission, including shapes with the same thread
count. The immutable requirement follows the definition through generated
artifacts and caches. Raw CUDA can declare the same contract through
`KernelLaunchRequirements(requiredBlock = Some(...))`; its author remains
responsible for matching that declaration to the handwritten source. This is
not automatic shape inference, a device-limit check, or a CUDA launch bound.

For DSL kernels, the normalizer also specializes `blockDim.x/y/z` from this
contract before CUDA generation. With the shape above, `blockDim.x * literal(2)`
becomes `256`, and comparisons against those known dimensions can select a
branch. Thread indices stay dynamic. Kernels without a required shape and
handwritten raw CUDA are not specialized. The typed launch check remains
essential: bypassing it can violate the assumptions of specialized code.

- **Launches:** Typed launches are asynchronous on CUDA's default stream or an
  owned explicit stream. `CudaContext.synchronize()` waits for all context
  work, while `CudaStream.synchronize()` waits for one explicit stream; launch
  never synchronizes implicitly.
- **Buffer ownership:** Owned `CudaDeviceBuffer` arguments must belong to the
  function's context. A mismatch returns `BufferContextMismatch` with a
  zero-based argument index before address packing or resource retention.
- **Copies:** Device-buffer copies are whole-buffer and synchronous through
  temporary pageable direct staging. A successful upload completes device DMA
  before returning, so a subsequent nonblocking-stream launch can consume it.
  Reusable `CudaPinnedBuffer[T]` storage
  provides a page-locked synchronous path without repeated direct-buffer
  allocation; its same-context device-buffer transfers also support explicit-stream
  `cuMemcpyHtoDAsync` and `cuMemcpyDtoHAsync` operations.
- **Events and dependencies:** `CudaEvent.record`, `query`, and `synchronize`
  provide completion markers, while `CudaStream.waitFor` establishes GPU-side
  stream dependencies. Pinned/device source and destination ranges are
  independently validated.
- **In-flight work:** Asynchronous copies and launches retain participating
  pinned buffers, device buffers, and modules through stream, event, or context
  completion. Explicit-stream work completes through stream, event, or context
  completion; default-stream launches complete through context synchronization.
  Closing an in-flight resource makes it unavailable immediately and defers
  native release.
- **Teardown:** Pinned host reads and writes are rejected during transfers. A
  stream synchronizes tracked work before destruction, and a context waits for
  pending default-stream launches before native teardown.

### Diagnostics and source locations

NVRTC diagnostics retain generated CUDA locations and map them
to the closest known Scala `SourceSpan` while preserving the original compiler
log. Scala 3 call-site capture now populates spans for DSL declarations,
stores, accumulation, structured control flow, reductions, and barriers.
Fine-grained expression/operator spans remain a later increment.

### Compilation artifacts and caches

Header-dependent kernels (for example, `Float16` with `cuda_fp16.h`) need an
explicit toolkit include directory when NVRTC has no configured header search
path. Pass it when generating the compilation artifact:

```scala
import flight4s.core.codegen.{CompilerOptions, CudaCodegen}
val include = java.nio.file.Path.of(sys.env("CUDA_PATH"), "include")
val options = CompilerOptions(additionalNvrtcOptions = Vector(s"--include-path=$include"))
val generated = CudaCodegen.generate(kernelDefinition, options)
```

`CUDA_PATH` is read by this application snippet, not implicitly by the compiler.
Cache identity includes options but not filesystem header contents; invalidate
affected caches if external headers change at the same path.

- **Identity:** A versioned canonical SHA-256 covers CUDA source and its
  generated/raw provenance, resolved options, target, compiler version,
  generated-source codegen version, program name, and kernel ABI/launch metadata.
  Encoding v3 includes optional exact block dimensions. Previous encoding-v2
  entries become cache misses. Codegen v26 supports signed grid dimensions,
  following v25's block specialization and v24's signed indexing repair;
  the store remains schema v2.
- **Memory:** `NvrtcCompilationCache` is a caller-owned bounded in-memory LRU
  of successful PTX compilations. Cache hits rebind PTX metadata to the current
  typed compilation input.
- **Persistent artifacts:** `NvrtcCompiler.version()` queries the loaded
  compiler without creating or compiling a program, so it participates in the
  key before lookup. The versioned, checksum-verified `NvrtcArtifactStore`
  atomically persists generated or raw CUDA C++, PTX, compiler logs, and
  compilation metadata. Reload validates provenance and compilation identity
  before rebinding to the caller's typed input. Generated diagnostics use the
  current Scala source map; raw diagnostics retain CUDA locations.
- **Persistent mode:** `NvrtcCompilationCache.persistent(maximumEntries,
  store)` composes memory, disk, then NVRTC. Store I/O failures are cache
  misses; invalid entries are removed and repaired by successful recompilation.
  `clear()` clears memory only, while `clearPersistent()` clears the disk layer.
  Manifest schema v2 stores source in `source.cu` and records provenance.
  Schema v1 entries are treated as incompatible cache entries and rebuilt on
  successful compilation.

### Raw CUDA interoperability

`flight4s.core.unsafe.raw.RawCuda.kernel` accepts handwritten CUDA C++ with an
explicit typed signature, compiler options, and launch requirements. Compile
with `NvrtcCompiler.compile(raw, target)`, load the artifact with
`context.load(artifact)`, resolve with `module.function(raw)`, and submit
`function.launch(raw.bind(arguments), config, stream)`.

Raw definitions must be the ones retained by the current artifact. Cache hits
rebind to the current caller's definition; native PTX sharing does not allow
cross-artifact or generated/raw invocation substitution. CUDA source/ABI
agreement remains the caller's responsibility. Raw code does not pass through
the DSL validator or IR optimizer.

`CudaFunction.kernel` exposes source-neutral entry-point, signature, and launch
metadata, replacing the pre-alpha `generated` accessor. Original generated or
raw source remains available through `function.module.artifact.input`.

### Native module reuse

Within one `CudaContext`, `load` also
deduplicates live native CUDA modules by the SHA-256 identity of their PTX.
Each caller receives its own provenance-aware `CudaModule` wrapper, while the
shared native module unloads only after the final wrapper and its in-flight work
release. Idle modules are not retained after the final close.

## Runnable Examples

The [row softmax example](examples/src/main/scala/flight4s/examples/RowSoftmax.scala)
contains the complete Scala kernel and `main`, including NVRTC compilation,
owned buffers, an explicit stream, synchronization, and cleanup.

Inspect generated CUDA without a native library or GPU:

```shell
sbt "examples/runMain flight4s.examples.RowSoftmax --cuda-source"
```

Run on CUDA after building the native library below:

```shell
sbt -Dflight4s.cuda.native.path=<absolute-library-path> "examples/runMain flight4s.examples.RowSoftmax"
```

It accepts finite row-major Float logits and uses one serial row per thread.
This is an end-to-end correctness example, not a performance-tuned softmax.
See [examples/README.md](examples/README.md) for contracts and test commands.

The [block-cooperative softmax](examples/src/main/scala/flight4s/examples/BlockRowSoftmax.scala)
uses one 128-thread block per row, strided functional ranges, and explicit
shared-memory reduction trees. It has a complete `main` and GPU numerical and
sanitizer coverage; it is not yet a performance-tuned inference operator.

The [row statistics example](examples/src/main/scala/flight4s/examples/RowStatistics.scala)
uses a pure Scala update function and `(count, mean, M2)` tuple fold to compute
Double means and population variances from Float rows. Values are widened before
arithmetic. It includes a complete `main`, high-precision CPU-reference tests,
and source inspection:

```shell
sbt "examples/runMain flight4s.examples.RowStatistics --cuda-source"
sbt -Dflight4s.cuda.native.path=<absolute-library-path> "examples/runMain flight4s.examples.RowStatistics"
```

[BlockRowStatistics](examples/src/main/scala/flight4s/examples/BlockRowStatistics.scala)
keeps that serial baseline intact and combines per-thread moments with an
explicit shared-memory tree. Its exact 128-thread block contract, empty-state
handling, and barriers are verified by GPU fixtures and CUDA sanitizers.

[RowLayerNorm](examples/src/main/scala/flight4s/examples/RowLayerNorm.scala)
combines a tuple Welford fold with a second functional traversal for per-column
gain and bias. Statistics and normalization use Double; the output is explicitly
narrowed to Float. It includes a complete `main`, positive-epsilon/input
validation, independent numerical tests, and inspectable generated CUDA.

[BlockRowLayerNorm](examples/src/main/scala/flight4s/examples/BlockRowLayerNorm.scala)
uses one exact 128-thread block per row, the existing shared-memory moment
merge, and a strided output traversal. It retains the serial example's host
contract; floating-point grouping differs. Numerical tests and memcheck,
racecheck, and synccheck verify the cooperative implementation.

## Build

Flight4s requires JDK 17 or newer and sbt:

```shell
sbt test
```

The optional native CUDA launcher requires CMake, a C++20 compiler (GCC 10+,
Clang 10+, or Visual Studio 2022+), JDK headers, and CUDA Toolkit 12 or newer:

```shell
cmake -S native -B native/build -DBUILD_TESTING=ON
cmake --build native/build --config Release
ctest --test-dir native/build -C Release --output-on-failure
```

See [native/README.md](native/README.md) for JNI and GPU integration tests.
The half-header GPU test additionally requires `CUDA_PATH` to identify a toolkit
containing `include/cuda_fp16.h`.

## Coordinates

The planned core artifact is:

```scala
"io.github.gpu-flight" %% "flight4s-core" % "<version>"
```

No release has been published yet.

## License

Flight4s is licensed under the terms in [LICENSE](LICENSE).
