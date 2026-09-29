package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

private case class ResultState(a: Expr[Int], b: Expr[Int]) derives ProductFoldState

class AutomaticResultSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
  private val out = output[Int]("out")
  private val signature = params(out)
  private def build(body: BlockBuilder ?=> Unit) = kernel("results", signature)(_ => body)
  private def key(expression: Expr[?]): String = expression match
    case Load(local: LocalVariable[?], _) => local.name
    case _ => fail("expected a result snapshot")
  private def same[Args <: Tuple](actual: Kernel[Args], expected: Kernel[Args]): Unit =
    assertEquals(actual.ir, expected.ir)
    assertEquals(KernelValidator.validate(actual), KernelValidator.validate(expected))
    assert(KernelValidator.validate(actual).isValid)
    assertEquals(EffectAnalysis.block(actual.body), EffectAnalysis.block(expected.body))
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected))

  test("folds and ordered sums accept unnamed result bindings"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.*
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      case class State(a: Expr[Int], b: Expr[Int]) derives ProductFoldState
      kernel("folds") {
        val range = gpuRange(literal(0), literal(4))
        val scalar = range.foldLeft(literal(0))(_ + _)
        val pair = range.foldLeft((literal(1), literal(2)))((s, _) => (s._2, s._1))
        val tuple = range.foldLeft((literal(1), literal(2), literal(3)))((s, _) => (s._2, s._3, s._1))
        val product = range.foldLeft(State(literal(1), literal(2)))((s, _) => State(s.b, s.a))
        val total = range.filter(_ > literal(1)).orderedSum(literal(0))
        ()
      }
    """).map(_.message), Nil)

  test("unnamed scalar folds delegate to every existing traversal implementation with identical IR"):
    val factories: Vector[() => GpuTraversal[Int]] = Vector(
      () => gpuRange("i", literal(0), literal(4)),
      () => gpuRange("i", literal(0), literal(4)).map(_ * literal(2)),
      () => gpuRange("i", literal(0), literal(4)).filter(_ > literal(1)),
      () => gpuRange("i", literal(0), literal(4)).flatMap(i => gpuRange("j", literal(0), i)),
      () => gpuRange("i", literal(0), literal(4)).map(i => (i, i + literal(1))).map(t => t._1 + t._2),
      () => gpuRange("i", literal(0), literal(4)).map(i => ResultState(i, i + literal(1))).map(s => s.a + s.b)
    )
    def check[Value](factory: () => GpuValueTraversal[Value])(step: (Expr[Int], Value) => Expr[Int]): Unit =
      var name = ""
      val actual = build {
        val result = factory().foldLeft(literal(5))(step)
        name = key(result)
        out(literal(0)) := result
      }
      val expected = build { out(literal(0)) := factory().foldLeft(name, literal(5))(step) }
      same(actual, expected)
    factories.foreach(factory => check(factory)(_ - _))
    check(() => gpuRange("i", literal(0), literal(4)).map(i => (i, i + literal(1))))((s, t) => s - t._1)
    check(() => gpuRange("i", literal(0), literal(4)).map(i => ResultState(i, i + literal(1))))((s, p) => s - p.a)

  test("pair tuple and product folds preserve simultaneous updates and exact named IR"):
    var prefix = ""
    val pair = build {
      val result = gpuRange("i", literal(0), literal(3)).foldLeft((literal(1), literal(2)))((s, _) => (s._2, s._1))
      prefix = key(result._1).stripSuffix("_0")
      out(literal(0)) := result._1 + result._2
    }
    same(pair, build {
      val result = gpuRange("i", literal(0), literal(3)).foldLeft(prefix, (literal(1), literal(2)))((s, _) => (s._2, s._1))
      out(literal(0)) := result._1 + result._2
    })
    val tuple = build {
      val result = gpuRange("i", literal(0), literal(3)).foldLeft((literal(1), literal(2), literal(3)))((s, _) => (s._2, s._3, s._1))
      prefix = key(result._1).stripSuffix("_0")
      out(literal(0)) := result._1 + result._3
    }
    same(tuple, build {
      val result = gpuRange("i", literal(0), literal(3)).foldLeft(prefix, (literal(1), literal(2), literal(3)))((s, _) => (s._2, s._3, s._1))
      out(literal(0)) := result._1 + result._3
    })
    val product = build {
      val result = gpuRange("i", literal(0), literal(3)).foldLeft(ResultState(literal(1), literal(2)))((s, _) => ResultState(s.b, s.a))
      prefix = key(result.a).stripSuffix("_0")
      out(literal(0)) := result.a + result.b
    }
    same(product, build {
      val result = gpuRange("i", literal(0), literal(3)).foldLeft(prefix, ResultState(literal(1), literal(2)))((s, _) => ResultState(s.b, s.a))
      out(literal(0)) := result.a + result.b
    })

  test("tuple and product source traversals support unnamed scalar and structured state"):
    val definition = build {
      val tuples = gpuRange("i", literal(0), literal(3)).map(i => (i, i + literal(1)))
      out(literal(0)) := tuples.foldLeft(literal(0))((s, t) => s + t._1)
      val pair = tuples.foldLeft((literal(0), literal(0)))((s, t) => (s._1 + t._1, s._2 + t._2))
      val products = gpuRange("j", literal(0), literal(3)).map(i => ResultState(i, i))
      out(literal(1)) := products.foldLeft(literal(0))((s, p) => s + p.a)
      val state = products.foldLeft(ResultState(literal(0), literal(0)))((s, p) => ResultState(s.a + p.a, s.b + p.b))
      out(literal(2)) := pair._1 + state.b
    }
    assert(KernelValidator.validate(definition).isValid)
    assert(CudaCodegen.generate(definition).isRight)

  test("orderedSum retains named sum IR without changing expression-only mapped sum"):
    var name = ""
    def traversal = gpuRange("i", literal(0), literal(4)).filter(_ > literal(0))
      .flatMap(i => gpuRange("j", literal(0), i)).map(convert.i32ToF32(_))
    val actual = build {
      val result = traversal.orderedSum(literal(0f))
      name = key(result)
      ()
    }
    same(actual, build { traversal.sum(name, literal(0f)); () })
    val expression = gpuRange("i", literal(0), literal(4)).map(identity).sum(literal(0))
    assert(expression.isInstanceOf[ReduceSum[?, ?]])
    val pure = build { out(literal(0)) := choose(literal(true))(expression)(literal(0)) }
    assertEquals(pure.body.statements.size, 1)
    assert(CudaCodegen.generate(pure).isRight)

  test("warp unnamed overloads preserve named operands widths masks and result snapshots"):
    val mask = literal(UInt.fromBits(-1))
    val x = threadIdx.x
    var keys = Vector.empty[String]
    val actual = build {
      keys = Vector(
        warp.shuffle(mask, x, literal(0)),
        warp.shuffleUp(mask, x, literal(UInt.fromBits(1)), 16),
        warp.shuffleDown(mask, x, literal(UInt.fromBits(1))),
        warp.shuffleXor(mask, x, literal(1), 8),
        warp.ballot(mask, x < literal(16)), warp.all(mask, x < literal(32)), warp.any(mask, x === literal(0)),
        warp.reduceSum(UInt.fromBits(-1), x), warp.reduceTree(UInt.fromBits(-1), x, 16)(_ - _)
      ).map(key)
    }
    same(actual, build {
      warp.shuffle(keys(0), mask, x, literal(0))
      warp.shuffleUp(keys(1), mask, x, literal(UInt.fromBits(1)), 16)
      warp.shuffleDown(keys(2), mask, x, literal(UInt.fromBits(1)))
      warp.shuffleXor(keys(3), mask, x, literal(1), 8)
      warp.ballot(keys(4), mask, x < literal(16))
      warp.all(keys(5), mask, x < literal(32))
      warp.any(keys(6), mask, x === literal(0))
      warp.reduceSum(keys(7), UInt.fromBits(-1), x)
      warp.reduceTree(keys(8), UInt.fromBits(-1), x, 16)(_ - _)
      ()
    })

  test("atomic unnamed overloads retain every named operation order scope and effect"):
    import MemoryOrder.*
    var keys = Vector.empty[String]
    val ref = atomic.device(out(literal(0)))
    val actual = build {
      keys = Vector(ref.load(Acquire), ref.exchange(literal(3), Relaxed),
        ref.fetchAdd(literal(1), AcquireRelease), ref.fetchSub(literal(2), Release),
        ref.fetchMin(literal(4), Relaxed), ref.fetchMax(literal(5), Relaxed),
        ref.fetchAnd(literal(7), Relaxed), ref.fetchOr(literal(8), Relaxed), ref.fetchXor(literal(9), Relaxed),
        ref.compareExchange(literal(1), literal(2), SequentiallyConsistent, Acquire),
        atomicFetchAdd(out(literal(0)), literal(1))).map(key)
    }
    same(actual, build {
      ref.load(keys(0), Acquire)
      ref.exchange(keys(1), literal(3), Relaxed)
      ref.fetchAdd(keys(2), literal(1), AcquireRelease)
      ref.fetchSub(keys(3), literal(2), Release)
      ref.fetchMin(keys(4), literal(4), Relaxed)
      ref.fetchMax(keys(5), literal(5), Relaxed)
      ref.fetchAnd(keys(6), literal(7), Relaxed)
      ref.fetchOr(keys(7), literal(8), Relaxed)
      ref.fetchXor(keys(8), literal(9), Relaxed)
      ref.compareExchange(keys(9), literal(1), literal(2), SequentiallyConsistent, Acquire)
      atomicFetchAdd(keys(10), out(literal(0)), literal(1))
      ()
    })
    assert(EffectAnalysis.block(actual.body).hasMemoryOrdering)

  test("unnamed result operations remain forbidden in expression-only callbacks"):
    val actions: Vector[BlockBuilder ?=> Expr[Int]] = Vector(
      gpuRange("i", literal(0), literal(2)).foldLeft(literal(0))(_ + _),
      gpuRange("i", literal(0), literal(2)).orderedSum(literal(0)),
      warp.shuffle(literal(UInt.fromBits(-1)), threadIdx.x, literal(0)),
      atomic.device(out(literal(0))).load(MemoryOrder.Acquire)
    )
    actions.foreach { action =>
      val error = intercept[DslError] { build { choose(literal(true))(action)(literal(0)); () } }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)
    }

  test("invalid atomic orders and warp masks remain rejected with caller spans"):
    val span = SourceSpan("Results.scala", 10, 2, 10, 30)
    val invalid = build {
      given DslSourcePosition = DslSourcePosition(span)
      atomic.device(out(literal(0))).load(MemoryOrder.Release)
      ()
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(e => (e.code, e.span)),
      Vector((ValidationCode.InvalidAtomicOrder, span)))
    val error = intercept[DslError] { build { warp.reduceSum(UInt.fromBits(0), threadIdx.x); () } }
    assertEquals(error.code, DslErrorCode.InvalidWarpReductionGroup)

  test("missing orders invalid state and unsupported atomic types do not compile"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad", params(output[Int]("out"))) { p => atomic.device(p._1(literal(0))).fetchAdd(literal(1)); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.MemoryOrder
      kernel("bad", params(output[Float]("out"))) { p => atomic.device(p._1(literal(0))).fetchMin(literal(1f), MemoryOrder.Relaxed); () }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("bad") { gpuRange(literal(0), literal(4)).foldLeft(0)(_ + _); () }
    """).nonEmpty)

  test("warp operations accept unnamed result bindings"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.UInt
      kernel("warpResults") {
        val mask = literal(UInt.fromBits(-1))
        val x = threadIdx.x
        warp.shuffle(mask, x, literal(0))
        warp.shuffleUp(mask, x, literal(UInt.fromBits(1)), width = 16)
        warp.shuffleDown(mask, x, literal(UInt.fromBits(1)))
        warp.shuffleXor(mask, x, literal(1))
        warp.ballot(mask, x < literal(16))
        warp.all(mask, x < literal(32))
        warp.any(mask, x === literal(0))
        warp.reduceSum(UInt.fromBits(-1), x)
        warp.reduceTree(UInt.fromBits(-1), x, width = 16)(_ - _)
        ()
      }
    """).map(_.message), Nil)

  test("atomic operations accept unnamed snapshots but retain explicit order"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.MemoryOrder.*
      kernel("atomicResults", params(inOut[Int]("data"))) { p =>
        val ref = atomic.device(p._1(literal(0)))
        ref.load(Acquire)
        ref.exchange(literal(1), Relaxed)
        ref.fetchAdd(literal(1), Relaxed)
        ref.fetchSub(literal(1), Relaxed)
        ref.fetchMin(literal(1), Relaxed)
        ref.fetchMax(literal(1), Relaxed)
        ref.fetchAnd(literal(1), Relaxed)
        ref.fetchOr(literal(1), Relaxed)
        ref.fetchXor(literal(1), Relaxed)
        ref.compareExchange(literal(1), literal(2), AcquireRelease, Acquire)
        atomicFetchAdd(p._1(literal(0)), literal(1))
        ()
      }
    """).map(_.message), Nil)
