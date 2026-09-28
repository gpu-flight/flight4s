package flight4s.core.dsl

import munit.FunSuite
import java.util.concurrent.{CountDownLatch, Executors, TimeUnit}

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.codegen.CudaCodegen
import flight4s.core.ir.*

class ExpressionStagingSuite extends FunSuite:
  test("reduction callbacks cannot append stores to a captured outer builder"):
    val out = output[Int]("out")
    val error = intercept[DslError] {
      kernel("badReduction", params(out)) { _ =>
        out(literal(0)) := reduceSum("i", literal(0), literal(4), literal(0)) { _ =>
          out(literal(1)) := literal(99)
          literal(1)
        }
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("conditional arms cannot stage unconditional stores"):
    val out = output[Int]("out")
    val error = intercept[DslError] {
      kernel("badConditional", params(out)) { _ =>
        out(literal(0)) := choose(literal(true))({
          out(literal(1)) := literal(99)
          literal(1)
        })(literal(2))
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("mapped foreach callbacks cannot move barriers outside the loop"):
    val error = intercept[DslError] {
      kernel("badMapping") {
        gpuRange("i", literal(0), literal(4)).map { i =>
          barrier()
          i
        }.foreach(_ => ())
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("mapped sums and fold steps cannot emit captured-builder effects"):
    val sumError = intercept[DslError] {
      kernel("badMappedSum") {
        gpuRange("i", literal(0), literal(4)).map { i => barrier(); i }.sum(literal(0))
        ()
      }
    }
    assertEquals(sumError.code, DslErrorCode.StatementInsideExpression)
    val foldError = intercept[DslError] {
      kernel("badFoldStep") {
        gpuRange("i", literal(0), literal(4)).foldLeft("acc", literal(0)) { (acc, i) =>
          barrier()
          acc + i
        }
        ()
      }
    }
    assertEquals(foldError.code, DslErrorCode.StatementInsideExpression)

  test("expression callbacks cannot hide declarations or statement-producing terminals"):
    val localError = intercept[DslError] {
      kernel("badExpressionLocal") {
        reduceSum("i", literal(0), literal(4), literal(0)) { _ => local("hidden", literal(1)).read }
        ()
      }
    }
    assertEquals(localError.code, DslErrorCode.StatementInsideExpression)
    assert(localError.span != SourceSpan.Unknown)
    val sharedError = intercept[DslError] {
      kernel("badExpressionShared") {
        reduceSum("i", literal(0), literal(4), literal(0)) { _ =>
          sharedArray[Int]("hidden", 4)
          literal(1)
        }
        ()
      }
    }
    assertEquals(sharedError.code, DslErrorCode.StatementInsideExpression)
    val terminalError = intercept[DslError] {
      kernel("badHiddenFold") {
        choose(literal(true))(
          gpuRange("i", literal(0), literal(4)).foldLeft("hidden", literal(0))(_ + _)
        )(literal(0))
        ()
      }
    }
    assertEquals(terminalError.code, DslErrorCode.StatementInsideExpression)

  test("both conditional arms stage once even when the condition is a literal"):
    var trueStages = 0
    var falseStages = 0
    val expression = choose(literal(true))({ trueStages += 1; literal(1) })({ falseStages += 1; literal(2) })
    assertEquals((trueStages, falseStages), (1, 1))
    assert(expression.isInstanceOf[Conditional[?]])
    val error = intercept[DslError] {
      kernel("badUnselectedArm") {
        choose(literal(true))(literal(1))({ barrier(); literal(2) })
        ()
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("guards restore after exceptions and reject before appending the offending statement"):
    val out = output[Int]("out")
    val definition = kernel("recoveredStaging", params(out)) { _ =>
      intercept[DslError] {
        choose(literal(true))({ barrier(); literal(1) })(literal(2))
      }
      intercept[IllegalArgumentException] {
        reduceSum("i", literal(0), literal(4), literal(0)) { _ =>
          throw IllegalArgumentException("host callback failed")
        }
      }
      out(literal(0)) := literal(7)
      barrier()
    }
    assertEquals(definition.body.statements.size, 2)
    assert(definition.body.statements.head.isInstanceOf[Store[?, ?]])
    assert(definition.body.statements(1).isInstanceOf[Barrier])
    assert(KernelValidator.validate(definition).isValid)

  test("nested expressions stay guarded while legitimate statement bodies remain available"):
    val error = intercept[DslError] {
      kernel("nestedGuard") {
        reduceSum("i", literal(0), literal(4), literal(0)) { _ =>
          choose(literal(true))(literal(1))(literal(2))
          barrier()
          literal(3)
        }
        ()
      }
    }
    assertEquals(error.code, DslErrorCode.StatementInsideExpression)
    val out = output[Int]("out")
    val valid = kernel("validExpressions", params(out)) { _ =>
      val total = reduceSum("i", literal(0), literal(2), literal(0)) { i =>
        reduceSum("j", literal(0), literal(2), literal(0))(j => choose(i < j)(i)(j))
      }
      out(literal(0)) := total
      gpuRange("k", literal(0), literal(2)).map(i => choose(i < literal(1))(i)(literal(0))).foreach { x =>
        scoped { out(x) := x; barrier() }
      }
    }
    assert(CudaCodegen.generate(valid).isRight)

  test("expression guards are isolated between concurrent kernel builders"):
    val entered = CountDownLatch(1)
    val release = CountDownLatch(1)
    val executor = Executors.newSingleThreadExecutor()
    try
      val request = executor.submit(() => reduceSum("i", literal(0), literal(2), literal(0)) { i =>
        entered.countDown()
        // MUnit evaluates assert conditions under the suite lock; never wait there on a worker.
        if !release.await(5, TimeUnit.SECONDS) then
          throw IllegalStateException("test did not release expression callback")
        i
      })
      assert(entered.await(5, TimeUnit.SECONDS), "expression callback did not start")
      val independent = kernel("independentBuilder") { local("x", literal(1)); barrier() }
      assert(KernelValidator.validate(independent).isValid)
      release.countDown()
      assert(request.get(5, TimeUnit.SECONDS).isInstanceOf[ReduceSum[?, ?]])
    finally
      release.countDown()
      executor.shutdownNow()
