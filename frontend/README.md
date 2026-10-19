# Scala Kernel Frontend Prototypes

## Quoted Scala Syntax Prototype

`ScalaKernel.kernel` is a separate inline quoted entry point, not another
rewrite of the annotation. Import its namespace instead of the explicit DSL's
wildcard in the source file using this frontend:

```scala
import scala.annotation.experimental
import flight4s.frontend.ScalaKernel.*

@experimental
def scale = kernel("scale", params(input[Float]("data"), output[Float]("target"),
    value[Int]("count"), value[Float]("factor"))) { (data, target, count, factor) =>
  val i = blockIdx.x * blockDim.x + threadIdx.x
  if i < count then
    val old = data(i)
    var total = old
    total += factor
    target(i) = if old < 0.0f then total else old
}
```

The factory returns `Kernel[(DeviceBuffer[Float], DeviceBuffer[Float], Int, Float)]`.
Separately compiled callers retain this exact launch type and opt in when using
an experimental factory. Imports do not require global experimental flags;
the kernel entry point is annotated, not its namespace. Do not add `@kernel`
to these factories: the existing annotation has a different contract below.

`threadIdx`/`blockIdx`/`blockDim`/`gridDim` expose all three axes as `Int`.
Mapped scalar parameters expose their primitive types; buffer parameters expose
typed `DeviceArray[T, Mode]` markers whose reads return `T`. These markers are
source syntax only, never JVM reads of GPU values or host arrays. Calling a
marker outside the captured body throws. The macro generates existing IR
construction code and does not retain or execute the source lambda.

Supported: literal lambdas with named tuple parameters or `p._N`, explicit
typed signatures (including empty), `Int`/`Float`/`Double`/`Boolean` literals and
parameters, arithmetic `+ - * /`, Int `% & | ^`, numeric comparisons,
primitive `== !=`, unary `+`, Int unary `-`, Boolean `! && ||`, primitive
`val` snapshots, initialized `var`, assignment/compound assignment, immutable
buffer aliases, static 1-D shared arrays and aliases, `sync.block()` and `sync.blockAfter`
(with `barrier()` retained for compatibility),
statement scopes, statement `if`/`else`, and pure expression
`if`/`else`, direct unit-stride `start until end` range loops with lazy guards,
and explicit `deviceRange` scalar/flat-tuple yield/map/flatMap/withFilter/foreach plans with ordered
scalar and immutable flat-tuple `foldLeft` initializers, including named tuples. Short-circuit
booleans lower to lazy `Conditional` IR. An input's
`update` requires read-write evidence, and lowering independently requires an
output parameter. Existing validation and CUDA generation remain authoritative.

Deferred/rejected: other loops, eager range `.filter`, `to`/`by`, stored range values,
eager Range `yield`, `match`/`try`/return, lazy/uninitialized bindings,
local definitions, general expression blocks, arbitrary tuple/product local state, source-level
embedded folds, case-class fold state/traversals and collectives, captured host state/values/helper calls,
numeric conversions, shifts, floating remainder, dynamic floating unary minus,
vector/low-precision source types, signature inference, and Unit-style annotated
methods. Dynamic floating negation is not approximated by `0 - x`, which would
mishandle signed zero. Negative floating literals remain supported. This is a
documented compiler subset, not a sandbox. Host kernel names and signatures
evaluate once in normal call order; that configuration is outside the body.

`ScalaKernelCompilerSuite` contains 29 actual compiler-program tests,
including separate callers, negative admission, all intrinsic axes, supported
operators, scopes and configuration evaluation order. `ScalaKernelIrSuite`
contains thirty-four tests: independent explicit-DSL branch/short-circuit/loop/guard/traversal/fold/flatMap/signature/shared-memory/phase references
compare exact IR/effects/generated artifacts after aligning only source spans;
all thirty-five fixtures validate and retain source maps and deterministic CUDA.
`ScalaKernelTupleCompilerSuite` adds twelve real compiler-program tests for flat
tuple admission, field types, tupled callbacks, nested plans and rejected effects.
`TupleSignatureCompilerSuite` retains seventeen signature compiler-program tests.
`ScalaKernelTupleFoldCompilerSuite` adds thirteen real compiler-program tests for
typed tuple states, seed/result aliases, dependent fields and rejected effects.
`ScalaKernelNamedTupleCompilerSuite` adds fourteen compiler-program tests for named
fields, mixed states, nesting, separate callers and rejected malformed/effectful inputs.
Four named fixtures share independent primitive-DSL and CPU references with their
unnamed counterparts, proving the field-label change preserves operations and order.
`ScalaKernelSharedCompilerSuite` adds twelve compiler/admission and host-marker
tests for shared types, aliases, placements, sizes, purity and divergence warnings.
Three independent shared-memory IR/GPU reference pairs cover cross-warp exchange,
repeated reuse, Boolean/Double arrays and rejected mismatched block launches.
`ScalaKernelPhaseCompilerSuite` adds sixteen admission/marker tests for exact
body-then-barrier order, scopes, nested/empty phases, purity and divergence,
including standalone sync imports, compatibility and the retired prototype spelling.
Two phase fixtures share unchanged CPU assertions with their explicit-barrier
counterparts and have separate independent IR/effect/generated-artifact references.
`ScalaKernelCudaJniSuite` has thirty-five real GPU fixtures for Float scaling, mutable
Int snapshots/shadowing, tightly sized short-circuit inputs, and Double/Boolean
branches. They cover counts 0/1/63/64/65/193/257, 32 output tails, both stream
paths, disabled execution, unchanged read-only inputs, and in-place updates.
The loop fixtures additionally cover Float row sums with 0/1/3/17/65 columns,
mutable bound snapshots, negative/empty/reversed ranges and signed Int edges,
dependent nested bounds, shadowed loop indices and per-iteration snapshots.
The guarded fixtures additionally check chained predicates protecting tight
allocations, all/none/mixed matches, live mutable state, guarded nested generators
and distinct/shadowed callback indices.
Traversal fixtures verify tight guarded rows, aliases/repeated terminals, live
captures after changed bounds, single-evaluation mapped values after buffer stores,
and nested lexical plans. Compiler cases cover all four scalar map result types,
literal-union widening, unused-plan purity and separate typed callers.
Fold fixtures verify Float cancellation order and seeds, filtered/empty traversals,
Int plan reuse after changed memory/bias, and nested Boolean/Double state with
distinct Int elements. Full buffers, tails and both stream paths are checked.
FlatMap fixtures additionally verify guarded inner-bound loads, per-outer bounds,
saved outer values across stores, reused inner aliases, three-level flattening,
live terminal captures and global Boolean/Double fold state.

### Shared Memory And Barrier Contract

`sharedArray[T](elementCount)` returns a `DeviceSharedArray[T]` marker, distinct
from the global `DeviceArray[T, Mode]` used by kernel arguments. It must directly
initialize an immutable val. Elements are `Int`, `Float`, `Double` or `Boolean`;
the size must be a positive compile-time Int constant in the typed tree. A normal
device val, parameter or host capture is not a compile-time size.

`sharedArray2D[T](rows, columns)` and `sharedArray3D[T](depth, rows, columns)`
return rank-2 and rank-3 markers addressed by logical indices, so
`tile(row, column)` lowers to a two-index `SharedElement` and emits
`tile[row][column]`. Every dimension is a positive compile-time Int constant,
and the declarations follow the same immutability, alias, scope and element-type
rules as rank 1. The physical row stride equals `columns`; the explicit DSL's
`rowStride` padding overloads are deliberately deferred in the quoted subset,
so declare wider `columns` and leave the padding cells unused.

`dynamicSharedArray[T]()` returns a sizeless rank-1 marker; the launch supplies
`dynamicSharedMemoryBytes` on `LaunchConfig`, and the runtime validates it
against the kernel's recorded element size and alignment before launch
(missing, undersized or misaligned sizes are structured failures). Declaration
and access rules match the static rank-1 marker; existing core validation
rejects a second dynamic declaration in the same kernel.

Declarations are admitted in the kernel body and unconditional lexical scopes,
including the wrapper Scala emits for tupled parameters. They register static
storage with the root kernel builder, while aliases remain lexically scoped.
New declarations inside branches, loops, traversal callbacks or expressions are
rejected, even when unused. Existing immutable aliases may be introduced inside
statement branches/loops. Aliases share storage; they do not copy array elements.
Actual marker symbols identify operations; similarly named host methods and
effectful module receivers are rejected. Markers throw outside captured code.

Reads and writes lower to existing `Load(SharedElement(...))` and `Store` IR.
They retain the Shared address space and source spans through effects, validation,
optimization and CUDA generation. Shared reads may occur in pure maps, guards and
folds; writes and barriers cannot hide in expression callbacks, even unused plans.
No buffer parameter, launch ABI, backend version, runtime or native change is needed.

`sync.block()` emits one `Barrier` at its exact statement position, producing
`__syncthreads()`. There is no implicit entry/exit barrier or JVM synchronization.
The earlier `barrier()` spelling remains supported with the same semantics.
See the complete [Scala/CUDA exchange example](../README.md#shared-memory-and-barriers).
For repeated cooperative reuse, synchronize after reads before overwriting:

```scala
for round <- deviceRange(0, rounds) do
  val previous = tile((lane + 1) % 64)
  sync.block()
  tile(lane) = previous + round
  sync.block()
```

Equivalent CUDA C++ after an initial tile fill and block barrier:

```cpp
const int start = 0, end = rounds;
for (int round = start; round < end; ++round) {
    const int previous = tile[(lane + 1) % 64];
    __syncthreads();
    tile[lane] = previous + round;
    __syncthreads();
}
```

Threads must participate uniformly in each block barrier; a lane-dependent branch
or loop can be unsafe. Existing `BarrierMayDiverge` warnings retain source locations
but are not a complete proof. The frontend does not initialize storage, insert
barriers, prove bounds/capacity/race safety, or infer launch geometry from array size.
Use an explicit `requiringBlock` contract when an algorithm requires a fixed shape.

Deferred: `rowStride` padding overloads, local/constant arrays, fences,
split arrival/wait and collective operations.
Existing explicit-DSL APIs are unchanged. The quoted `sync.blockAfter { body }` helper below means body followed
by one block barrier, not mutual exclusion.
Other synchronization functions require their own participation, scope and ordering
contracts rather than automatically sharing that wrapper.

### Block Synchronization Contract

`ScalaKernel.sync.blockAfter(body: => Unit)` is statement syntax for one lexical
device scope followed by one block barrier. The macro translates the body once
through the existing statement path, emits `ScopedBlock(body)`, then appends
`Barrier` to the enclosing block. Each participating thread executes that body
once per dynamic visit. Both emitted nodes retain the phase call's source span;
body statements retain their own locations. There is no new IR or native API.
See the [Scala/CUDA example](../README.md#block-synchronization).

Naming: `sync.block()` synchronizes at that statement; `sync.blockAfter { ... }`
executes the body first. `block` here names the CUDA thread scope, not the lexical
braces. The unreleased `block.phase` prototype is replaced, not retained as an
alias. Existing quoted `barrier()` and all explicit-DSL APIs remain unchanged.
`sync.warp(mask)` and `sync.warpAfter(mask) { ... }` are implemented with their
own compile-time mask and participation contract; see the warp section below.

The by-name marker never evaluates its argument on the JVM, including when an
out-of-capture call throws. Direct, qualified and renamed imported API symbols
are supported. Host lookalikes and effectful module receivers remain rejected.
The body uses the same bounded statement subset, not an arbitrary stored callback.
Phase results cannot initialize device vals or enter pure map/guard/fold callbacks.
Body locals do not escape; enclosing vars and shared aliases remain usable.
Unconditional phase scopes retain root shared-declaration ownership. A phase
inside a branch/loop cannot bypass the existing declaration restriction.

Empty phases still emit a barrier. Nested phases each add their own trailing
barrier, and explicit body barriers are preserved without deduplication. A branch
inside the body does not guard the trailing barrier, but a branch around the
whole phase does. Existing `BarrierMayDiverge` warnings report at the phase call;
no warning is not a complete participation or race-safety proof.

A phase has no entry barrier, lock, implicit initialization, return value or
cleanup/finally semantics. Existing return/throw/try rejection is unchanged.
It does not infer launch shape or guarantee safety within the body. For example,
after an initial tile fill and synchronization, reuse still needs a read barrier:

```scala
for round <- deviceRange(0, rounds) do
  sync.blockAfter {
    val previous = tile((lane + 1) % 64)
    sync.block()
    tile(lane) = previous + round
  }
```

Equivalent CUDA C++ (bounds and names simplified):

```cpp
for (int round = 0; round < rounds; ++round) {
    {
        const int previous = tile[(lane + 1) % 64];
        __syncthreads();
        tile[lane] = previous + round;
    }
    __syncthreads();
}
```

The explicit inner barrier finishes all reads before overwrite; the trailing
phase barrier finishes all writes before the next round. Do not automatically
reuse this contract for memory fences or split arrival/wait. Those remain
separate frontend slices, as do dynamic/multidimensional shared arrays.

### Warp Synchronization Contract

`sync.warp(mask)` emits one warp barrier for exactly the lanes named by the mask.
`sync.warpAfter(mask) { body }` translates the body once into a lexical device
scope, then appends one warp barrier, mirroring `sync.blockAfter`. Both lower to
the existing core `WarpBarrier`; there is no new IR, native API or launch change.
The scope and trailing barrier keep the call-site span; body statements keep theirs.

The mask must be a compile-time constant Int: a literal such as `-1`, `0xffffffff`
or `0x0000ffff`, or the library constant `sync.fullMask` (all 32 lanes). The bit
pattern is taken as-is and emitted as CUDA `unsigned int`. An empty mask is rejected
at compile time. Dynamic masks, host-computed masks, captures and general UInt
expressions are rejected; they are a separate future contract.

```scala
sync.warpAfter(sync.fullMask) {
  val item = if i < count then data(i) else 0.0f
  tile(lane) = item
}
if i < count then target(i) = tile((lane + 1) % 32)
```

Equivalent CUDA C++ (names simplified):

```cpp
{
    const float item = i < count ? data[i] : 0.0f;
    tile[lane] = item;
}
::__syncwarp(0xffffffffu);
if (i < count) target[i] = tile[(lane + 1) % 32];
```

A mask does not filter which lanes execute the body or the barrier. Every calling
lane must be named by the mask, and every named non-exited lane must reach the
same barrier with the same mask: the CUDA participation contract is the caller's
obligation. Divergent enclosing control reports `WarpParticipationMayDiverge` at
the call site as a diagnostic, not a proof. Disjoint constant masks may synchronize
independent lane groups under lane-parity branches. Like block phases, a warp
phase has no entry barrier, lock, lane guard, initialization or race repair, and
shared-memory reuse still needs an explicit read barrier inside the body. Empty
and nested phases retain every barrier without deduplication. Warp barriers order
only the named lanes; block-wide publication still requires a block barrier.

### Warp Reduction Contract

`warpReduceSum(mask, value)` and `warpReduceSum(mask, value, width)` stage the
existing explicit-DSL shuffle-tree reduction (`warp.reduceSum`): one
`__shfl_xor_sync` exchange per tree level plus order-preserving selects, over
independent aligned lane groups of the compile-time width (default 32). The
result must directly initialize a primitive `val` or `var`; embedded reductions
inside expressions or pure callbacks are rejected, exactly like `foldLeft`.
Values are Int, Float or Double. This is a parallel reduction tree, not an
ordered `foldLeft` sum: for floating-point values the two are not interchangeable.

The mask reuses the warp synchronization admission (compile-time constant Int
literals or `sync.fullMask`) and must additionally contain complete aligned
groups of the width; the width must be a literal 1, 2, 4, 8, 16 or 32. Both are
checked at compile time, mirroring the staged core validation. Every calling
lane must be named by the mask and participate; a mask never filters execution.

```scala
val item = if i < count then data(i) else 0.0f
val total = warpReduceSum(sync.fullMask, item)
if i < count then target(i) = total
```

Equivalent CUDA C++ (names simplified; one exchange per level):

```cpp
const float item = i < count ? data[i] : 0.0f;
float current = item;
for (int distance = 1; distance < 32; distance *= 2) {   // unrolled in practice
    const float partner = ::__shfl_xor_sync(0xffffffffu, current, distance, 32);
    current = current + partner;                          // lane-order-preserving selects elided
}
const float total = current;
if (i < count) target[i] = total;
```

With `width` 16 or 8 and a full mask, each aligned half or eighth of the warp
reduces independently and every lane reads its own group's result. Divergence
diagnostics report `WarpParticipationMayDiverge` per staged shuffle under
lane-varying control; block-uniform control stays quiet. No new core IR, native
behavior or launch change.

`warpReduceTree(mask, value)(combine)` and the `width` overload stage the same
shuffle tree with an explicit combine (`warp.reduceTree`). The combine must be
a literal two-parameter lambda over the value type, pure like every expression
callback: stores, barriers, captures and host calls are rejected. Subtree order
is preserved in every lane, so noncommutative combines are deterministic, and
`warpReduceTree(mask, value)(_ + _)` lowers identically to `warpReduceSum`.
Mask, width, value-type and initializer-only admission are shared with the sum
form.

```scala
val item = if i < count then data(i) else -1000.0f
val best = warpReduceTree(sync.fullMask, item)((a, b) => if a > b then a else b)
if i < count then target(i) = best
```

Equivalent CUDA C++ (names simplified; one exchange per level):

```cpp
const float item = i < count ? data[i] : -1000.0f;
float current = item;
for (int distance = 1; distance < 32; distance *= 2) {   // unrolled in practice
    const float partner = ::__shfl_xor_sync(0xffffffffu, current, distance, 32);
    const float left = lowerLane ? current : partner;     // subtree order per lane
    const float right = lowerLane ? partner : current;
    current = left > right ? left : right;
}
if (i < count) target[i] = current;
```

Votes remain a separate quoted slice.

### Warp Shuffle Contract

`warpShuffle`, `warpShuffleUp`, `warpShuffleDown` and `warpShuffleXor` stage the
four explicit-DSL shuffle primitives: register exchanges between the lanes of a
warp, with no shared memory, no barrier and no memory ordering. Masks and widths
reuse the warp admission (compile-time constant masks, literal widths 1 through
32; shuffles need no aligned-group rule, only a nonempty mask). Direct source
lanes admit any admitted Int expression, and a literal source must be
nonnegative; up/down deltas and xor lane masks are compile-time literals in
0 to 31, because core deltas are UInt and quoted code admits no conversions.
Values are Int, Float or Double; results are initializer-only.

```scala
var total = if i < count then data(i) else 0.0f
val add1 = warpShuffleUp(sync.fullMask, total, 1)
if lane >= 1 then total += add1
val add2 = warpShuffleUp(sync.fullMask, total, 2)
if lane >= 2 then total += add2
// ... distances 4, 8, 16 complete the inclusive prefix sum.
```

Equivalent CUDA C++ (names simplified; one step shown):

```cpp
float total = i < count ? data[i] : 0.0f;
const float add1 = ::__shfl_up_sync(0xffffffffu, total, 1u, 32);
if (lane >= 1) total = total + add1;
```

The width splits the warp into independent segments and lane movement wraps
within each segment; up/down lanes outside the segment keep their own value.
Every calling lane must be named by the mask, and reading from an inactive or
unnamed lane is undefined: `WarpParticipationMayDiverge` stays a diagnostic per
shuffle under lane-varying control. Dynamic deltas and lane masks, UInt values
and votes/ballots remain deferred.

### Block Reduction Contract

`blockReduction[T](threads)` declares power-of-two shared reduction scratch for
exactly `threads` lanes per 1-D block (a literal 1 through 1024), staging the
explicit-DSL `block.reduction`. Like `sharedArray`, the declaration owns shared
storage: it is immutable, root-scope-only, admits immutable aliases, and supports
Int, Float and Double elements. `scratch.sum(value)` and
`scratch.reduceTree(value)((a, b) => ...)` stage the strided scratch tree: every
thread stores its value, and one barrier follows the store, each combining level
and the final read, so the scratch is immediately reusable by the next call.
Results are initializer-only, exactly like the warp reductions, and the combine
is the same pure literal two-parameter lambda contract as `warpReduceTree`.

Each call records the declared block shape as a launch requirement: the kernel
must carry a matching `requiringBlock`, and validation rejects a mismatch. Every
thread of the block must reach every call; the staged barriers make divergent
calls unsafe, and `BarrierMayDiverge` reports one diagnostic per barrier under
lane-varying control while block-uniform control stays quiet.

```scala
val scratch = blockReduction[Float](256)
val item = if i < count then data(i) else 0.0f
val total = scratch.sum(item)
if i < count then target(i) = data(i) / total
```

Equivalent CUDA C++ (names simplified; the emitter unrolls the levels):

```cpp
__shared__ float scratch[256];
const float item = i < count ? data[i] : 0.0f;
float total = item;
scratch[rank] = total;                       // rank linearizes threadIdx
__syncthreads();
for (int distance = 1; distance < 256; distance *= 2) {
    if (rank % (distance * 2) == 0)
        scratch[rank] = scratch[rank] + scratch[rank + distance];
    __syncthreads();
}
total = scratch[0];
__syncthreads();                             // scratch reusable by the next call
if (i < count) target[i] = data[i] / total;
```

This is a tree combine over shared memory, not an ordered fold; for
floating-point values the two are not interchangeable. Multi-dimensional block
shapes, votes and shuffles remain separate quoted slices.

### Generic Tuple Signature Contract

`params(tuple)`, `CudaDsl.paramsTuple(tuple)` and `KernelSignature.fromTuple(tuple)`
are inline factories. They infer the tuple from the argument first, then resolve
`KernelParamTuple[Params]` with `summonInline`. This avoids the Scala 3.8.1
pre-expansion tree-check assertion caused by caller-side contextual evidence
inference when composing a generic tuple factory directly with the quoted builder.
The issue reproduced even with two parameters, independent of macro expansion.
Neither `-Xcheck-macros` nor `-Ycheck:all` is disabled; no toolchain upgrade is needed.

The exact `KernelArgumentsOf[Params]` and `Bindings = Params` contract is unchanged.
Bindings evaluate once. The runtime implementation, descriptor order, packed bytes
and native layout remain the same. Positional overloads through six parameters are
unchanged; larger signatures use one tuple, not an untyped varargs parameter list.
Empty tuples, `Tuple1` and arities 2/6/7/23 are checked by actual compiler programs,
including separate callers and a seven-parameter tupled lambda on the GPU.
`TupleSignatureCompilerSuite` adds 17 compiler-program tests covering all factory
paths, generic helpers, exact launch arity/order/types and read-only access rejection.

Source migration: generic inline factories no longer accept an explicit `(using tuple)`
argument. For low-level supplied evidence and an already typed binding tuple, use
`KernelSignature.fromTupleWithEvidence(bindings)(using tuple)`. Generic wrappers
with a given `KernelParamTuple[P]` can still call the ordinary inline factories;
the supplied given is found during expansion. This pre-release source API change
does not change the CUDA kernel ABI. Existing compiled callers should be rebuilt.

For Scala and equivalent CUDA C++, see the [tuple signature example](../README.md#tuple-signatures).
The existing six-parameter flatMap fixture is retained unchanged; the new
seven-parameter fixture exercises pointers, Int, Float, Boolean and Double arguments.

### Range Loop Contract

Inside a quoted kernel, ordinary constants need no `literal` wrapper:

```scala
var total = 0.0f
for column <- 0 until columns do
  val item = data(row * columns + column)
  total += item
target(row) = total
```

Equivalent CUDA C++ with simplified names:

```cpp
float total = 0.0f;
const int start = 0;
const int end = columns;
for (int column = start; column < end; ++column) {
    const float item = data[row * columns + column];
    total = total + item;
}
target[row] = total;
```

The typed tree is checked against the actual standard-library symbols for
`Predef.intWrapper`, `RichInt.until` and `Range.foreach`, not method spellings.
The macro constructs existing `ForLoop` IR through `gpuFor`, never a JVM Range.
Both bounds snapshot once, in start/end order, each time the range is entered.
This differs deliberately from explicitly passing a live mutable bound to
the existing `gpuFor` API, whose contract remains unchanged. A nested range
captures its bounds each outer iteration. Loop locals initialize each iteration;
enclosing vars update existing handles; same-name indices use lexical symbols.
Explicit nested loops, multiple-generator `for ... do`, and direct range
`.foreach` use this same lowering. No parallel reassociation is introduced.
Pattern generators, generator aliases, `yield`, inclusive/strided ranges,
stored/host-produced ranges and nonliteral callback values remain outside this
subset. Host-method eta expansion still fails when its body makes a host call.

### Lazy Range Guards

```scala
for column <- -1 until columns + 1 if column >= 0 if column < columns if data(row * columns + column) > threshold do
  val item = data(row * columns + column)
  total += item
```

Equivalent CUDA C++, with simplified names:

```cpp
const int start = -1;
const int end = columns + 1;
for (int column = start; column < end; ++column) {
    if (column >= 0) {
        if (column < columns) {
            if (data[row * columns + column] > threshold) {
                const float item = data[row * columns + column];
                total += item;
            }
        }
    }
}
```

Actual standard-library `withFilter`/filtered `foreach` symbols are recognized,
and the receiver chain must end at a direct supported `until` range. Each guard
becomes an existing `IfThen` inside that range's `ForLoop`. Later predicates,
body declarations and nested range bounds execute only when earlier guards
pass. Predicate and body lambda symbols bind independently to the same device
index, preserving distinct names and lexical shadows. Predicates read live
enclosing locals each iteration; only range bounds snapshot on entry.

Multiple guards, guarded nested/multiple generators, and direct
`(start until end).withFilter(i => condition).foreach { i => ... }` are supported.
Predicates use the same pure primitive-expression subset as other conditions:
no assignment/declaration blocks, host helpers/captures or stored callbacks.
Eager `.filter`, materialized/stored ranges, generator aliases and Range `yield` remain
deferred. No JVM collection/filter object or parallel collective is introduced.

### Staged Traversal Contract

`deviceRange(from, until)` explicitly chooses lazy GPU traversal semantics. It
is a marker returning `DeviceTraversal[Int]`, not a Scala Range or allocated buffer.
Its half-open unit-stride bounds snapshot once, in start/end order, when execution
reaches that device statement. Mappings and guards do not execute until a terminal.

```scala
val values = for column <- deviceRange(-1, columns + 1) if column >= 0 if column < columns
  yield data(row * columns + column)
values.withFilter(item => item > threshold).map(item => item * 2.0f)
  .foreach(item => total += item)
```

Equivalent CUDA C++ with simplified names:

```cpp
const int start = -1;
const int end = columns + 1;
for (int column = start; column < end; ++column) {
    if (column >= 0) {
        if (column < columns) {
            const float item = data[row * columns + column];
            if (item > threshold) {
                const float doubled = item * 2.0f;
                total += doubled;
            }
        }
    }
}
```

Immutable plan vals and aliases are reusable; each `foreach`/`for ... do` generates
a new serial loop over the same captured bounds. Mappings and guards read current
enclosing locals/device memory at that terminal and each visited index. Lexical
compiler symbols bind callbacks independently, including same-name shadows.
Every map result uses a per-item local snapshot before subsequent guards/maps/body.
Repeating or reading it after a terminal store never reevaluates that map. This
snapshot rule is stronger than plain expression-tree composition in the explicit
DSL; that DSL's existing behavior is unchanged.

Callbacks must be literal lambdas with pure supported primitive expressions.
Unused plans are checked too, but emit no mapping loop. No host lambda or traversal
object is retained, and all traversal markers throw outside capture. Recognized
symbols belong to `ScalaKernel`, so user-defined lookalike methods are not admitted.
The checked inferred map type is retained through lowering, including conditional
literal unions, without adding implicit numeric conversion.

Deferred: sums, case-class elements,
generator aliases/patterns, stored callbacks, mutable plans, filter/materialization,
inclusive/strided ranges, helpers/captures and expression blocks. Ordinary Scala
Range `map`/`yield` remains rejected: it is eager and cannot silently mean this lazy
plan. Existing core traversals, validation, effects, optimizers and CUDA backend
remain unchanged; only the quoted frontend constructs their existing IR nodes.

### Ordered Scalar Fold Contract

```scala
val values = for column <- deviceRange(0, columns) if column % 2 == 0
  yield data(row * columns + column)
val total = values.foldLeft(7.0f)((sum, item) => sum - item)
target(row) = total
```

Equivalent CUDA C++ (simplified names):

```cpp
const int start = 0, end = columns;
float accumulator = 7.0f;
for (int column = start; column < end; ++column) {
    if (column % 2 == 0) {
        const float item = data[row * columns + column];
        accumulator = accumulator - item;
    }
}
const float total = accumulator;
target[row] = total;
```

`DeviceTraversal[T].foldLeft[A](initial)(step)` must directly initialize a
primitive `val` or `var` in a statement body, including branches and nested loops.
Receiver construction comes first, including any new bound snapshots; the seed
snapshots once next, then the loop runs in ascending order. Guards suppress later
maps and updates. A pure literal `(state, item) => expression` reads old state;
the returned value replaces it once per accepted element. Empty/reversed/fully
filtered traversals return the seed. The source local snapshots the completed
result, so reused plans get independent accumulators and earlier results persist.

Elements may be supported primitives or flat primitive tuples. The accumulator
is primitive in this contract; its type may differ without an implicit conversion.
Immutable tuple states follow the [tuple fold contract](#tuple-fold-contract).
Source `var` results can be assigned afterward.
Captured device locals/memory remain live at each terminal, and existing per-item
map snapshots remain intact. Actual library symbols identify folds; no host step
lambda executes or survives construction. Original seed, step, terminal and local
spans are retained through existing local/ForLoop/Store/Load IR.

Folds in buffer stores, assignments, arithmetic, conditional expressions, seeds,
maps, predicates or fold steps remain rejected. Introducing statement-producing
folds into lazy expression contexts needs a separate evaluation-order contract.
Named product state, numeric conversions, helper/stored callbacks, general expression
blocks, sums, parallel trees and collectives are deferred; no reassociation or new
IR/backend/runtime behavior is implied.

### Nested Scalar Traversal Contract

```scala
val values = for
  outer <- deviceRange(0, rounds)
  if outer % 2 == 0
  inner <- deviceRange(0, outer)
  if inner > 0
yield outer + inner
val total = values.foldLeft(7)((sum, item) => sum - item)
```

Equivalent CUDA C++ (simplified names):

```cpp
const int start = 0, end = rounds;
int accumulator = 7;
for (int outer = start; outer < end; ++outer) {
    if (outer % 2 == 0) {
        const int innerStart = 0, innerEnd = outer;
        for (int inner = innerStart; inner < innerEnd; ++inner) {
            if (inner > 0) {
                const int item = outer + inner;
                accumulator = accumulator - item;
            }
        }
    }
}
const int total = accumulator;
```

`DeviceTraversal[T].flatMap[U](T => DeviceTraversal[U])` accepts a literal callback
returning a supported scalar or flat-tuple plan: a `deviceRange` chain or lexically bound plan
alias, optionally mapped/guarded/flattened. All input/result types remain supported
primitives or flat primitive tuples. Canonical inferred types are retained; unused nested callbacks are
recursively validated without emitting their device loops.

An existing root plan keeps its construction-time bounds. A new inner range
captures start/end once, in order, for each accepted outer element at each
terminal. Earlier failed guards suppress inner-bound reads entirely. Subsequent
stores can affect the next outer iteration, not the current inner loop's bounds.
An existing inner alias uses its original captured bounds, not new snapshots.
Outer map results remain saved across inner iterations and terminal writes.

Recursive lowering uses existing nested `ForLoop`, `IfThen`, local snapshots and
stores. Tail stages execute for each accepted inner item, including further
flattening; callback symbols preserve outer captures and same-name shadows.
`foreach` terminals may have statement bodies; flatMap/map/guard callbacks stay
pure, with no general declaration/assignment blocks or nested fold expressions.
Projection-only immutable tuple bindings introduced by tupled callbacks are admitted. One
ordered fold accumulator surrounds all nested loops, not one per inner range.
Empty/reversed/rejected inner plans perform no updates. Reusing a plan emits a
fresh nested traversal and independent fold state without retaining host lambdas,
objects or intermediate collections. No parallel reassociation is introduced.

Deferred: case-class elements/states, generator aliases/patterns, conditional
plan factories, stored/helper callbacks, captures, expression blocks and eager
Scala collection roots. The explicit DSL and annotation semantics are unchanged.
The generic tuple-signature checker issue found during fixture development is
resolved by staged evidence lookup; see the generic signature contract above.
The original six-parameter flatMap fixtures remain unchanged.

### Flat Tuple Traversal Contract

```scala
val pairs = deviceRange(0, count).map(i => (i, data(i)))
pairs.withFilter((index, item) => index % 2 == 0 && item > 0.0f)
  .foreach { (index, item) => target(index) = item * 2.0f }
val total = pairs.foldLeft(7.0f)((sum, pair) => sum - pair._2)
```

Equivalent CUDA C++ (simplified names; tupled `foreach` may introduce additional
primitive alias snapshots):

```cpp
const int start = 0, end = count;
for (int i = start; i < end; ++i) {
    const int index = i;
    const float item = data[i];
    if (index % 2 == 0 && item > 0.0f) {
        target[index] = item * 2.0f;
    }
}
float accumulator = 7.0f;
for (int i = start; i < end; ++i) {
    const int index = i;
    const float item = data[i];
    accumulator = accumulator - item;
}
const float total = accumulator;
```

Admission is a direct standard `Tuple1`..`Tuple22` constructor with only
`Int`/`Float`/`Double`/`Boolean` fields, or an existing traversal tuple binding.
Maps snapshot every field left to right, including forwarded/identity tuple
maps and unused fields of a consumed map. Earlier guards suppress the whole map;
later guards run after every field snapshot. A rejected later guard cannot erase
an earlier field read. Unused plans still validate all fields without emitting loops.

Fields use typed `_N` and literal `tuple(index)` access. Tupled callback projection
bindings and immutable aliases of existing tuple bindings are recognized without
admitting arbitrary host products or effects. Map/flatMap can change scalar/tuple
shape, and a tuple source can feed guards, foreach and a scalar or tuple-state fold.
Outer tuple snapshots survive inner iterations and stores; new inner bounds still
capture once per accepted outer item. Plan aliases/reuse retain the scalar contract.

No CUDA aggregate, new IR node, ABI descriptor, backend version or native code is
needed: the macro carries a compile-time list of typed field handles and emits
existing primitive locals/loads. This is stricter than the explicit DSL's plain
expression-tree tuple traversal: quoted maps automatically snapshot each field.

Deferred/rejected: empty/nested/host-valued tuples, tuples above 22 fields,
case-class products, arbitrary tuple-valued locals, mutable
tuple aliases, conditional tuple constructors, dynamic indexing, casts/productElement,
stored/helper callbacks, implicit numeric conversions, embedded folds and effectful
blocks. The 22-field boundary is a constructor-admission slice, not a CUDA or Scala
tuple limitation; larger tuples require separately tested compiler tree handling.
Kernel signature tuples and the explicit DSL continue supporting arities above 22.

### Tuple Fold Contract

```scala
val pairs = deviceRange(0, count).map(i => (i, data(i)))
val result = pairs.foldLeft((0.0f, 0)) { (state, pair) =>
  (state._1 + pair._2, state._2 + 1)
}
val saved = result
val next = pairs.foldLeft(saved)((state, pair) => (state._1 - pair._2, state._2 + 1))
target(0) = result._1 + next._1
```

The first fold lowers to scalar CUDA state (simplified names):

```cpp
float sum = 0.0f;
int visits = 0;
for (int i = start; i < end; ++i) {
    const int index = i;
    const float item = data[i];
    const float nextSum = sum + item;
    const int nextVisits = visits + 1;
    sum = nextSum;
    visits = nextVisits;
}
```

`foldLeft` must directly initialize an immutable tuple `val`. State is a flat,
nonempty standard `Tuple1`..`Tuple22` of `Int`/`Float`/`Double`/`Boolean` fields.
Seeds and steps use direct standard tuple constructors or existing tuple bindings,
including an earlier fold result or a traversal element. A step returning `state`
is valid. Pure conditional expressions may occur within individual fields.

Receiver bounds are captured before seed fields, which evaluate once left to right.
Each accepted traversal element computes all next fields against the previous
state, snapshots them left to right, then assigns accumulator fields in order.
The accumulator surrounds the complete flatMap nest. An empty or fully rejected
traversal returns its seed. Ordered arithmetic is never reassociated implicitly.

The completed accumulator fields become an immutable result binding. Aliases use
these stable fields; a later fold snapshots its seed into separate locals, so it
cannot modify earlier results. Projections keep exact field types and may feed
later scalar expressions, maps and folds. Every seed and step field is validated,
even when the result is unused or the range is empty.

No tuple object, aggregate IR, CUDA struct or ABI type is introduced. Existing
scalar folds retain their original IR. Mutable tuple results/aliases, arbitrary
tuple locals, case-class/nested products, tuples above 22 fields, conditional tuple
constructors, pattern bindings, helper/stored callbacks, host captures, effects,
conversions and embedded fold expressions remain rejected. Projection-only
immutable callback aliases follow the existing tuple traversal rule.

### Named Tuple Contract

The quoted frontend also accepts flat Scala 3 named tuples:

```scala
val items = deviceRange(0, count).map(i => (index = i, value = data(i)))
val stats = items.foldLeft((sum = 0.0f, count = 0)) { (state, item) =>
  (sum = state.sum + item.value, count = state.count + 1)
}
target(0) = stats.sum
```

Equivalent CUDA structure, with readable names in place of generated identifiers:

```cpp
float sum = 0.0f;
int visits = 0;
for (int i = 0; i < count; ++i) {
    const int index = i;
    const float value = data[i];
    const float nextSum = sum + value;
    const int nextVisits = visits + 1;
    sum = nextSum;
    visits = nextVisits;
}
target[0] = sum;
```

The 1..22-field primitive tuple rules apply unchanged to named traversal elements
and immutable fold states. Names and field order remain Scala type information;
they do not name CUDA locals. Type aliases such as `type Stats = (sum: Float,
count: Int)` may be declared outside the kernel body. Named field selections and
literal `state(0)` projections are supported. Identity steps/maps, immutable
aliases, projection-only callback vals, earlier-result seeds, ordered guards,
mixed named/unnamed maps and nested flatMap all retain the existing snapshot rules.

The macro recognizes the actual standard-library `NamedTuple` type, `build` and
`apply` symbols. It translates their inline argument proxies, not stale call-trace
symbols, so nested callbacks retain lexical binding identity. Every name must be
a distinct literal string and the label count must equal the value count, even
for manually encoded `NamedTuple` aliases. No host tuple is constructed at staging
time and no new IR, CUDA aggregate, launch ABI or compiler option is introduced.

General named-tuple locals, mutable states, nested fields, dynamic indices,
whole-tuple conditional expressions, `toTuple`/other tuple transformations and
pattern bindings remain outside this slice. Construct a supported mapped tuple
explicitly when changing shape. Case-class constructors are not admitted: the
pinned macro setup exposes symbol trees without constructor bodies, so this
frontend cannot safely erase their potential effects. The explicit DSL's
`ProductFoldState` still uses its existing host-staging contract. A separately
validated case-class schema is future work, not an implicit fallback.

## Kernel Annotation Prototype

This optional Scala 3.8.1 module implements `flight4s.frontend.kernel` using
Scala's experimental `MacroAnnotation` API. It annotates an existing typed
kernel factory and rewrites `Expr[T]` vals in statement-producing bodies to
`CudaDsl.let` snapshots, including supported nested DSL callbacks. Initialized
`Expr[T]` vars lower to typed device locals and assignments to device stores.

## Public Contract

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

The factory's existing method and `Kernel[Args]` return type remain visible in
separately compiled Scala source. No generated sibling method is required.
Factories are concrete, non-inline, have no type/value/context parameters,
and directly call `CudaDsl.kernel` with a literal body lambda. An empty `()`
parameter list is allowed. Kernel names and typed signatures stay explicit.
`input`/`output` are the canonical spellings; their short aliases are available
in the explicit DSL. Output buffers are read-write, not write-only.

A separately compiled caller retains the exact launch argument types:

```scala
import scala.annotation.experimental
import flight4s.core.ir.{DeviceBuffer, Kernel}

@experimental
def definition: Kernel[(DeviceBuffer[Float], DeviceBuffer[Float], DeviceBuffer[Float], Int)] =
  vectorAdd
```

Both definition and caller explicitly opt in. Scala's `-experimental` flag
marks all top-level definitions experimental and propagates to consumers;
the project does not enable that flag. The stable core/runtime/examples use
their existing compiler settings. Development tests enable `-Xcheck-macros`
and `-Ycheck:all`, including actual separate compiler invocations.

## Snapshot Semantics

Inside the direct kernel body or a supported nested statement body:

```scala
val original = data(i).read
data(i) := literal(9)
target(i) := original
```

The annotation inserts the equivalent of `let(data(i).read)` at the val
declaration. The later use reads the saved device value. The initializer
constructs its expression once, and the original val's Scala source span is
retained. Each factory invocation builds a fresh kernel with automatic local
binding identities; generated CUDA names remain deterministic.

Explicit `let` vals are preserved without an extra snapshot. Nested `when`,
both `gpuIf` alternatives, `scoped`, `gpuFor`, and traversal `foreach` bodies
use the callback's own `BlockBuilder`. This includes mapped, tuple-valued,
filtered, and flattened traversal terminals.

```scala
when(i < count) {
  val original = data(i).read
  data(i) := literal(9)
  target(i) := original
}
```

The snapshot stays inside the branch; an inactive lane does not load `data(i)`.
A snapshot inside a loop executes again on each device iteration. Nested
callbacks may read earlier snapshots from enclosing lexical scopes.

Expression-only `map`, filter predicates, reduction expressions, and fold
steps do not provide a statement builder for implicit snapshots. Initialized
`Expr` vals there are rejected; callback parameters are not declarations.
Explicit attempts to emit DSL statements there retain the core's
`DslErrorCode.StatementInsideExpression` staging rejection. This slice does
not support implicit snapshots in arbitrary Scala expression blocks.

## Mutable Device Locals

Inside an annotated statement body:

```scala
var total = literal(0.0f)
gpuRange(literal(0), columns).foreach { column =>
  val before = total
  total = total + data(row * columns + column).read
  previous(row) := before
}
target(row) := total
```

The annotation replaces the Scala var with an immutable hidden
`LocalVariable[Float]` handle. Each read becomes a `Load` and each assignment a
`Store`, using the existing typed IR, effects, validation, optimizers, and codegen.
The explicit DSL equivalent uses `local(literal(0.0f))`, `.read`, and `:=`.
The initializer constructs its expression once and runs at its lexical CUDA
declaration, including again on each iteration for a loop-local declaration.
There is no host-side assignment to an Expr reference.

Bindings must have exactly the public `Expr[T]` type, inferred or annotated;
use `literal(0.0f)`, not a host `0.0f`. Assignments must be statements in the
direct body or supported nested callbacks. Enclosing variables can be updated
from branches, scopes, and foreach/for bodies. Compiler-symbol identity keeps
same-named nested variables distinct. Compound assignment such as `+=` follows
Scala's existing assignment desugaring. `val before = total` remains a snapshot
even after subsequent assignments, whereas reading `total` sees the current value.
An explicit `let` used as a var initializer retains its own snapshot plus the
separate writable local; it is not an alias to mutable storage.

Pure expression callbacks may read an enclosing device local under the existing
IR read-effect rules, but cannot declare vars or assign them. General Scala
expression blocks with assignments are not admitted. A statement callback
embedded inside a pure callback still raises `StatementInsideExpression` during
staging; the annotation does not bypass that boundary. Declarations, assignments,
and rewritten reads retain their original source spans.

## Supported Subset

- Immutable bindings with exactly the public `Expr[T]` type in the direct
  body and supported nested statement callbacks.
- Initialized mutable `Expr[T]` bindings, reads, and statement assignments in
  those same bodies, including compound assignment after Scala desugaring.
- Literal host constants and existing non-Expr core DSL binding objects as metadata.
- Existing DSL operations, typed buffer access, arithmetic and explicit
  `when`/`gpuIf` control flow, subject to existing IR validation.
- The complete existing CUDA C++ -> NVRTC/PTX -> typed launch pipeline.

The prototype rejects host vars, `lazy val`, ordinary control flow, host assignment,
implicit Expr vals in expression-only contexts, arbitrary host/helper calls (including
parameterless methods), captured mutable host state, captured external Expr
values, tuple/product snapshot bindings, and concrete IR-node subtype vals.
The restriction is a documented compiler subset, not a security sandbox.
The method's signature and kernel name are normal host-side DSL construction.

The annotation runs after Scala type checking. It cannot make ordinary
`if (Expr[Boolean])` type-check. A Unit-returning CUDA-looking method with typed
parameters and automatic signature inference remain future frontend work.
Ordinary device control flow is implemented in the
separate quoted subset above, not by changing this annotation's Expr types.

## Verification

From the repository root, with the JNI library built:

```powershell
java "-Dsbt.supershell=false" "-Dflight4s.cuda.native.path=C:\Users\myoun\Documents\sources\gpuflight\flight4s\native\build\Release\flight4s_cuda.dll" -jar C:\Users\myoun\AppData\Local\Temp\sbt-launch-1.12.2.jar "frontendTests/test"
```

`KernelAnnotationSuite` compiles definition/use programs separately, checks
opt-in and typed argument admission, and runs negative programs through actual
compiler phases. `AnnotationIrSuite` compares exact IR, effects, validation,
generated artifacts (including explicit-let branch/loop references),
deterministic names, lexical placement, and source mapping. The existing GPU tests
run vector addition and Int/Float snapshot-after-write cases through NVRTC on
both default and explicit streams, including bounds, sentinels, and input
preservation. Without the native property, GPU fixtures skip.

Nested GPU cases cover zero/partial work, both branch alternatives, enclosing
scope aliases, and zero/one/multiple loop iterations. A mapped/filtered/flattened
`foreach` verifies that snapshots refresh rather than hoist. Compiler fixtures
also retain default-argument getter symbols when copying method selections.
Mutable-state tests compare exact branch/shadow/loop IR and generated artifacts
with explicit local/read/store references. Eight GPU fixtures also cover Int
branch updates, same-name scope isolation, loop-carried state, per-iteration
locals, Float foreach row sums, saved previous values, empty work, and tails.
Compiler programs verify scalar/vector types, purity and host-state rejection,
and initialization with default-argument calls.

The compiler harness depends on the pinned compiler only in the non-published
test project. The published frontend depends on the stable core.

## References

- [NVIDIA thread-block synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#thread-block-synchronization-functions)
- [Scala named tuples](https://docs.scala-lang.org/scala3/reference/other-new-features/named-tuples.html)
- [Scala experimental definitions](https://docs.scala-lang.org/scala3/reference/other-new-features/experimental-defs.html)
- [Scala quoted code](https://docs.scala-lang.org/scala3/guides/macros/quotes.html)
- [Scala macro reflection](https://docs.scala-lang.org/scala3/guides/macros/reflection.html)
- [Scala 3 quoted reflection](https://docs.scala-lang.org/scala3/reference/metaprogramming/reflection.html)
- [Pinned Scala 3.8.1 annotation API](https://www.scala-lang.org/api/3.8.1/scala/annotation.html)

The source contract was also read from `org.scala-lang:scala-library:3.8.1:sources`,
archive entry `scala/annotation/MacroAnnotation.scala`.
