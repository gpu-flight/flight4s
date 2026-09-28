package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*
import flight4s.core.codegen.CudaCodegen

class ScopedAtomicSuite extends FunSuite:
  test("typed atomic references expose explicitly ordered named results and stores"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.{Expr, MemoryOrder}
      kernel("atomics", params(output[Int]("counter"), output[Double]("sum"))) { p =>
        val counter = atomic.device(p._1(literal(0)))
        val ticket: Expr[Int] = counter.fetchAdd("ticket", literal(1), MemoryOrder.Relaxed)
        val before: Expr[Int] = counter.compareExchange("before", literal(1), literal(2),
          MemoryOrder.AcquireRelease, MemoryOrder.Acquire)
        counter.store(literal(3), MemoryOrder.Release)
        val current: Expr[Int] = counter.load("current", MemoryOrder.Acquire)
        val previous: Expr[Double] = atomic.device(p._2(literal(0))).exchange("previous", literal(1.0), MemoryOrder.SequentiallyConsistent)
        val oldSum: Expr[Double] = atomic.device(p._2(literal(0))).fetchAdd("oldSum", literal(2.0), MemoryOrder.Relaxed)
      }
    """), Nil)

  test("load store and compare-exchange orders have an explicit supported matrix"):
    for order <- MemoryOrder.values do
      val load = kernel("load", params(output[Int]("counter"))) { p =>
        atomic.device(p._1(literal(0))).load("seen", order); ()
      }
      val store = kernel("store", params(output[Int]("counter"))) { p =>
        atomic.device(p._1(literal(0))).store(literal(1), order)
      }
      assertEquals(KernelValidator.validate(load).isValid,
        Set(MemoryOrder.Relaxed, MemoryOrder.Acquire, MemoryOrder.SequentiallyConsistent).contains(order))
      assertEquals(KernelValidator.validate(store).isValid,
        Set(MemoryOrder.Relaxed, MemoryOrder.Release, MemoryOrder.SequentiallyConsistent).contains(order))
      for failure <- MemoryOrder.values do
        val compare = kernel("compare", params(output[Int]("counter"))) { p =>
          atomic.device(p._1(literal(0))).compareExchange("seen", literal(0), literal(1), order, failure); ()
        }
        val allowed = order match
          case MemoryOrder.Relaxed | MemoryOrder.Release => Set(MemoryOrder.Relaxed)
          case MemoryOrder.Acquire | MemoryOrder.AcquireRelease => Set(MemoryOrder.Relaxed, MemoryOrder.Acquire)
          case MemoryOrder.SequentiallyConsistent => Set(MemoryOrder.Relaxed, MemoryOrder.Acquire, MemoryOrder.SequentiallyConsistent)
        val result = KernelValidator.validate(compare)
        assertEquals(result.isValid, allowed.contains(failure), s"$order/$failure")
        if !result.isValid then
          assertEquals(result.errors.map(_.code), Vector(ValidationCode.InvalidAtomicOrder))
          assertNotEquals(result.errors.head.span, SourceSpan.Unknown)
          assert(CudaCodegen.generate(compare).isLeft)

  test("all RMW operations retain exact order scope result identity and operand count"):
    for order <- MemoryOrder.values do
      val definition = kernel("updates", params(output[Int]("counter"))) { p =>
        val ref = atomic.block(p._1(literal(0)))
        ref.exchange("exchange", literal(2), order)
        ref.fetchAdd("add", literal(3), order)
        ref.fetchSub("sub", literal(4), order)
        ref.fetchMin("minimum", literal(5), order)
        ref.fetchMax("maximum", literal(6), order)
        ref.fetchAnd("oldAnd", literal(7), order)
        ref.fetchOr("oldOr", literal(8), order)
        ref.fetchXor("oldXor", literal(9), order)
        ()
      }
      val results = definition.body.statements.collect { case x: AtomicResult[?, ?] => x }
      assertEquals(results.map(_.operation), Vector(AtomicOperation.Exchange, AtomicOperation.FetchAdd,
        AtomicOperation.FetchSub, AtomicOperation.FetchMin, AtomicOperation.FetchMax, AtomicOperation.FetchAnd,
        AtomicOperation.FetchOr, AtomicOperation.FetchXor))
      assert(results.forall(x => x.order == order && x.scope == AtomicScope.Block && x.operands.size == 1))
      assertEquals(KernelValidator.validate(definition).errors, Vector.empty)
      assertEquals(IrNormalizer.kernel(definition.ir).body, definition.body)
      val code = CudaCodegen.generate(definition).toOption.get.cudaSource
      assert(code.contains(order.cudaName), code)
      assert(code.contains(AtomicScope.Block.cudaName), code)

  test("unsupported storage access modes types and address spaces are rejected statically"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad", params(input[Int]("input"))) { p => atomic.device(p._1(literal(0))); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { atomic.block(local("x", literal(1))); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { atomic.device(sharedArray[Int]("x", 1)(literal(0))); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.Float4
      kernel("bad", params(output[Float4]("out"))) { p => atomic.device(p._1(literal(0))); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.MemoryOrder
      kernel("bad", params(output[Float]("out"))) { p =>
        atomic.device(p._1(literal(0))).fetchMin("old", literal(1f), MemoryOrder.Relaxed); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.MemoryOrder
      kernel("bad", params(output[Double]("out"))) { p =>
        atomic.device(p._1(literal(0))).compareExchange("old", literal(1.0), literal(2.0),
          MemoryOrder.Relaxed, MemoryOrder.Relaxed); () }
    """).nonEmpty)

  test("raw IR still validates arity capability result scope and failure metadata"):
    val definition = kernel("raw", params(output[Int]("counter"))) { p =>
      atomic.device(p._1(literal(0))).fetchAdd("old", literal(1), MemoryOrder.Relaxed); ()
    }
    val original = definition.body.statements.head.asInstanceOf[AtomicResult[Int, Global]]
    val invalid = Vector(
      original.copy(operands = Vector.empty) -> ValidationCode.InvalidAtomicOperands,
      original.copy(failureOrder = Some(MemoryOrder.Relaxed)) -> ValidationCode.InvalidAtomicOrder,
      original.copy(operation = AtomicOperation.CompareExchange, operands = Vector(literal(0), literal(1))) -> ValidationCode.InvalidAtomicOrder,
      AtomicResult(LocalVariable("old", I32), LocalVariable("notMemory", I32), AtomicOperation.Load,
        Vector.empty, I32, MemoryOrder.Relaxed, AtomicScope.Block) -> ValidationCode.InvalidAtomicAddressSpace,
      AtomicResult(LocalVariable("old", I32), SharedElement("scratch", Vector(literal(0)), I32), AtomicOperation.Load,
        Vector.empty, I32, MemoryOrder.Relaxed, AtomicScope.Device) -> ValidationCode.InvalidAtomicScope,
      AtomicResult(LocalVariable("old", F32), BufferElement[Float, ReadWrite]("floatCounter", literal(0), F32), AtomicOperation.FetchMin,
        Vector(literal(1f)), F32, MemoryOrder.Relaxed, AtomicScope.Device) -> ValidationCode.UnsupportedAtomicOperation
    )
    for (statement, code) <- invalid do
      val raw = Kernel(definition.ir.copy(body = Block(Vector(statement))))
      assert(KernelValidator.validate(raw).errors.exists(_.code == code), code.toString)
      assert(CudaCodegen.generate(raw).isLeft)

  test("ordered effects stay in statement order and repeated result reads do not repeat atomics"):
    val definition = kernel("snapshots", params(output[Int]("counter"), output[Int]("out"))) { p =>
      val ref = atomic.device(p._1(literal(0)))
      val before = ref.load("before", MemoryOrder.Acquire)
      ref.store(literal(9), MemoryOrder.Release)
      val after = ref.load("after", MemoryOrder.Acquire)
      p._2(literal(0)) := before + before
      p._2(literal(1)) := after
      when(after > literal(0)) { barrier() }
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(normalized.body.statements.take(3), definition.body.statements.take(3))
    assertEquals(EffectAnalysis.block(normalized.body), EffectAnalysis.block(definition.body))
    assert(EffectAnalysis.block(normalized.body).hasMemoryOrdering)
    val effects = definition.body.statements.take(3).map(EffectAnalysis.statement)
    assertEquals(effects(0).writtenSpaces, Set(EffectMemorySpace.Local))
    assertEquals(effects(1).readSpaces, Set.empty[EffectMemorySpace])
    assertEquals(KernelValidator.validate(definition).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))
    val code = CudaCodegen.generate(definition).toOption.get.cudaSource
    assertEquals("::__nv_atomic_load".r.findAllIn(code).size, 2)
    assert(code.contains("CUDA 12.8 or newer"))
    assert(code.contains("compute capability 6.0 or newer"))

  test("atomic result scopes ownership conflicts and expression guards are preserved"):
    val target = output[Int]("counter")
    val ref = atomic.device(target(literal(0)))
    val missing = kernel("missing") { ref.load("old", MemoryOrder.Relaxed); () }
    assert(KernelValidator.validate(missing).errors.nonEmpty)
    val collision = kernel("collision", params(target)) { _ =>
      ref.load("old", MemoryOrder.Relaxed)
      ref.load("old", MemoryOrder.Relaxed)
      ()
    }
    assert(KernelValidator.validate(collision).errors.exists(_.code == ValidationCode.DuplicateLocalName))
    val error = intercept[DslError] {
      kernel("hidden", params(target)) { _ =>
        choose(literal(true))(ref.load("old", MemoryOrder.Acquire))(literal(0)); ()
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("legacy atomicAdd emission is unchanged and does not gain scoped toolchain guards"):
    val definition = kernel("legacy", params(output[Int]("counter"))) { p =>
      atomicAdd(p._1(literal(0)), literal(1))
    }
    assertEquals(CudaCodegen.generate(definition).toOption.get.cudaSource,
      "extern \"C\" __global__ void legacy(int* counter) {\n  ::atomicAdd(&counter[0], 1);\n}\n")

  test("C++20 keywords and alternative operator tokens fail DSL validation before NVRTC"):
    val keywords = "alignas alignof asm auto bool break case catch char char8_t char16_t char32_t class concept const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept nullptr operator private protected public register reinterpret_cast requires return short signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while and and_eq bitand bitor compl not not_eq or or_eq xor xor_eq".split(" ").toVector
    for name <- keywords do
      val definition = kernel("keyword", params(output[Int]("counter"))) { p =>
        atomic.device(p._1(literal(0))).load(name, MemoryOrder.Relaxed); ()
      }
      assert(KernelValidator.validate(definition).errors.exists(_.code == ValidationCode.InvalidLocalName), name)
      assert(CudaCodegen.generate(definition).isLeft, name)
      assert(KernelValidator.validate(kernel(name) { () }).errors.exists(_.code == ValidationCode.InvalidKernelName), name)
      assert(ModuleValidator.validate(module(constants = Vector(constantArray[Int](name, 1)))).errors.nonEmpty, name)
