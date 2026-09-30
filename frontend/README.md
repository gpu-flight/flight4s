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
buffer aliases, statement scopes, statement `if`/`else`, and pure expression
`if`/`else`, and direct unit-stride `start until end` range loops with lazy guards. Short-circuit
booleans lower to lazy `Conditional` IR. An input's
`update` requires read-write evidence, and lowering independently requires an
output parameter. Existing validation and CUDA generation remain authoritative.

Deferred/rejected: other loops, eager range `.filter`, `to`/`by`, stored range values,
`yield`, `match`/`try`/return, lazy/uninitialized bindings,
local definitions, expression blocks, tuple/product local state, source-level
functional traversals and collectives, captured host state/values/helper calls,
numeric conversions, shifts, floating remainder, dynamic floating unary minus,
vector/low-precision source types, signature inference, and Unit-style annotated
methods. Dynamic floating negation is not approximated by `0 - x`, which would
mishandle signed zero. Negative floating literals remain supported. This is a
documented compiler subset, not a sandbox. Host kernel names and signatures
evaluate once in normal call order; that configuration is outside the body.

`ScalaKernelCompilerSuite` contains 20 actual compiler-program tests,
including separate callers, negative admission, all intrinsic axes, supported
operators, scopes and configuration evaluation order. `ScalaKernelIrSuite`
contains nine tests: independent explicit-DSL branch/short-circuit/loop/guard references
compare exact IR/effects/generated artifacts after aligning only source spans;
all ten fixtures validate and retain source maps and deterministic CUDA.
`ScalaKernelCudaJniSuite` has ten real GPU fixtures for Float scaling, mutable
Int snapshots/shadowing, tightly sized short-circuit inputs, and Double/Boolean
branches. They cover counts 0/1/63/64/65/193/257, 32 output tails, both stream
paths, disabled execution, unchanged read-only inputs, and in-place updates.
The loop fixtures additionally cover Float row sums with 0/1/3/17/65 columns,
mutable bound snapshots, negative/empty/reversed ranges and signed Int edges,
dependent nested bounds, shadowed loop indices and per-iteration snapshots.
The guarded fixtures additionally check chained predicates protecting tight
allocations, all/none/mixed matches, live mutable state, guarded nested generators
and distinct/shadowed callback indices.

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
Eager `.filter`, materialized/stored ranges, generator aliases and `yield` remain
deferred. No JVM collection/filter object or parallel collective is introduced.

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

- [Scala experimental definitions](https://docs.scala-lang.org/scala3/reference/other-new-features/experimental-defs.html)
- [Scala quoted code](https://docs.scala-lang.org/scala3/guides/macros/quotes.html)
- [Scala macro reflection](https://docs.scala-lang.org/scala3/guides/macros/reflection.html)
- [Scala 3 quoted reflection](https://docs.scala-lang.org/scala3/reference/metaprogramming/reflection.html)
- [Pinned Scala 3.8.1 annotation API](https://www.scala-lang.org/api/3.8.1/scala/annotation.html)

The source contract was also read from `org.scala-lang:scala-library:3.8.1:sources`,
archive entry `scala/annotation/MacroAnnotation.scala`.
