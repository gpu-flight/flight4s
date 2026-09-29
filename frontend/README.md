# Kernel Annotation Prototype

This optional Scala 3.8.1 module implements `flight4s.frontend.kernel` using
Scala's experimental `MacroAnnotation` API. It annotates an existing typed
kernel factory and rewrites `Expr[T]` vals in statement-producing bodies to
`CudaDsl.let` snapshots, including supported nested DSL callbacks.

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

## Supported Subset

- Immutable bindings with exactly the public `Expr[T]` type in the direct
  body and supported nested statement callbacks.
- Literal host constants and existing non-Expr core DSL binding objects as metadata.
- Existing DSL operations, typed buffer access, arithmetic and explicit
  `when`/`gpuIf` control flow, subject to existing IR validation.
- The complete existing CUDA C++ -> NVRTC/PTX -> typed launch pipeline.

The prototype rejects Scala `var`, `lazy val`, ordinary control flow/assignment,
implicit Expr vals in expression-only contexts, arbitrary host/helper calls (including
parameterless methods), captured mutable host state, captured external Expr
values, tuple/product snapshot bindings, and concrete IR-node subtype vals.
The restriction is a documented compiler subset, not a security sandbox.
The method's signature and kernel name are normal host-side DSL construction.

The annotation runs after Scala type checking. It cannot make ordinary
`if (Expr[Boolean])` type-check. A Unit-returning CUDA-looking method with typed
parameters, automatic signature inference, mutable state,
and ordinary device control flow remain future frontend work.

## Verification

From the repository root, with the JNI library built:

```powershell
java "-Dsbt.supershell=false" "-Dflight4s.cuda.native.path=C:\Users\myoun\Documents\sources\gpuflight\flight4s\native\build\Release\flight4s_cuda.dll" -jar C:\Users\myoun\AppData\Local\Temp\sbt-launch-1.12.2.jar "frontendTests/test"
```

`KernelAnnotationSuite` compiles definition/use programs separately, checks
opt-in and typed argument admission, and runs negative programs through actual
compiler phases. `AnnotationIrSuite` compares exact IR, effects, validation,
generated artifacts (including explicit-let branch/loop references),
deterministic names, lexical placement, and source mapping. Five GPU tests
run vector addition and Int/Float snapshot-after-write cases through NVRTC on
both default and explicit streams, including bounds, sentinels, and input
preservation. Without the native property, GPU fixtures skip.

Nested GPU cases cover zero/partial work, both branch alternatives, enclosing
scope aliases, and zero/one/multiple loop iterations. A mapped/filtered/flattened
`foreach` verifies that snapshots refresh rather than hoist. Compiler fixtures
also retain default-argument getter symbols when copying method selections.

The compiler harness depends on the pinned compiler only in the non-published
test project. The published frontend depends on the stable core.

## References

- [Scala experimental definitions](https://docs.scala-lang.org/scala3/reference/other-new-features/experimental-defs.html)
- [Scala 3 quoted reflection](https://docs.scala-lang.org/scala3/reference/metaprogramming/reflection.html)
- [Pinned Scala 3.8.1 annotation API](https://www.scala-lang.org/api/3.8.1/scala/annotation.html)

The source contract was also read from `org.scala-lang:scala-library:3.8.1:sources`,
archive entry `scala/annotation/MacroAnnotation.scala`.
