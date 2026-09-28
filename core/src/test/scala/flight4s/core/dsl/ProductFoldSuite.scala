package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

private case class Rotation(first: Expr[Int], second: Expr[Int], third: Expr[Int]) derives ProductFoldState
private case class MixedProduct(count: Expr[Int], total: Expr[Double], seen: Expr[Boolean]) derives ProductFoldState

class ProductFoldSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)
  test("case classes derive typed fold state with named expression fields and copy"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.dsl.ProductFoldState
      import flight4s.core.ir.Expr
      case class Moments(count: Expr[Int], total: Expr[Double], seen: Expr[Boolean]) derives ProductFoldState
      kernel("productState") {
        val result: Moments = gpuRange("i", literal(0), literal(8))
          .foldLeft("state", Moments(literal(0), literal(0.0), literal(false))) { (s, i) =>
            s.copy(count = s.count + literal(1), total = s.total + convert.i32ToF64(i), seen = s.seen || (i > literal(3)))
          }
        ()
      }
    """), Nil)

  test("product and tuple state emit identical IR and CUDA for every traversal kind"):
    val out = output[Int]("out")
    val signature = params(out)
    for kind <- Vector("range", "mapped", "filtered", "flatMapped") do
      def definition(product: Boolean) = kernel("rotation", signature) { _ =>
        val range = gpuRange("i", literal(0), literal(9)).by(2)
        val traversal: GpuTraversal[Int] = kind match
          case "range" => range
          case "mapped" => range.map(_ + literal(1))
          case "filtered" => range.filter(_ > literal(0))
          case _ => range.flatMap(i => gpuRange("j", literal(1), i).map(j => i / j))
        val result = if product then
          val state = traversal.foldLeft("state", Rotation(literal(1), literal(2), literal(3))) { (s, i) =>
            Rotation(s.second, s.third, s.first + i)
          }
          (state.first, state.second, state.third)
        else traversal.foldLeft("state", (literal(1), literal(2), literal(3))) { (s, i) =>
          (s._2, s._3, s._1 + i)
        }
        out(literal(0)) := result._1
        out(literal(1)) := result._2
        out(literal(2)) := result._3
      }
      val actual = definition(true)
      val expected = definition(false)
      assertEquals(actual.ir, expected.ir, kind)
      assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(expected), kind)

  test("mixed fields preserve source spans and snapshot every value before stores"):
    val span = SourceSpan("Product.scala", 18, 2, 18, 90)
    var result = Option.empty[MixedProduct]
    val definition = kernel("productSpans") {
      result = Some(gpuRange("i", literal(0), literal(4))
        .foldLeft("state", MixedProduct(literal(0), literal(0.0), literal(false))) { (s, i) =>
          s.copy(count = s.count + i, total = s.total + convert.i32ToF64(s.count), seen = s.count > literal(0))
        }(using summon[ProductFoldState[MixedProduct]], summon[BlockBuilder], DslSourcePosition(span)))
    }
    assertEquals(definition.body.statements.size, 4)
    assert(definition.body.statements.forall(_.span == span))
    val body = definition.body.statements.last.asInstanceOf[ForLoop].body.statements
    assertEquals(body.size, 6)
    assert(body.take(3).forall(_.isInstanceOf[LocalDeclaration[?]]))
    assert(body.drop(3).forall(_.isInstanceOf[Store[?, ?]]))
    assert(body.forall(_.span == span))
    assertEquals(List(result.get.count.span, result.get.total.span, result.get.seen.span), List.fill(3)(span))
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)

  test("product callbacks stage once even when the device range is empty"):
    var steps = 0
    val definition = kernel("emptyProduct") {
      gpuRange("i", literal(0), literal(0))
        .foldLeft("state", Rotation(literal(1), literal(2), literal(3))) { (s, _) => steps += 1; s }
      ()
    }
    assertEquals(steps, 1)
    assertEquals(KernelValidator.validate(definition).errors, Vector.empty)

  test("product steps and reconstructed constructors cannot emit hidden statements"):
    val stepError = intercept[DslError] {
      kernel("effectfulProduct") {
        gpuRange("i", literal(0), literal(4))
          .foldLeft("state", Rotation(literal(1), literal(2), literal(3))) { (s, _) => barrier(); s }
        ()
      }
    }
    assertEquals(stepError.code, DslErrorCode.StatementInsideExpression)
    val constructorError = intercept[DslError] {
      kernel("effectfulConstructor") {
        var emit = false
        case class Bad(value: Expr[Int]) derives ProductFoldState:
          if emit then barrier()
        val initial = Bad(literal(0))
        emit = true
        gpuRange("i", literal(0), literal(4)).foldLeft("state", initial)((s, _) => s)
        ()
      }
    }
    assertEquals(constructorError.code, DslErrorCode.StatementInsideExpression)
    assert(KernelValidator.validate(kernel("guardRestored") { barrier() }).isValid)

  test("product result fields remain subject to scope and generated-name validation"):
    val out = output[Int]("out")
    var escaped = Option.empty[Rotation]
    val invalid = kernel("escapedProduct", params(out)) { _ =>
      scoped {
        escaped = Some(gpuRange("i", literal(0), literal(4))
          .foldLeft("state", Rotation(literal(1), literal(2), literal(3)))((s, _) => s))
      }
      out(literal(0)) := escaped.get.first
    }
    assertEquals(KernelValidator.validate(invalid).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    val conflict = kernel("productConflict") {
      local("state_next_2", literal(0))
      gpuRange("i", literal(0), literal(4))
        .foldLeft("state", Rotation(literal(1), literal(2), literal(3)))((s, _) => s)
      ()
    }
    assert(!KernelValidator.validate(conflict).isValid)

  test("host fields empty products nested products and writable results are rejected"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.ProductFoldState
      case class Host(count: Int) derives ProductFoldState
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.ProductFoldState
      case class Empty() derives ProductFoldState
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.ProductFoldState
      import flight4s.core.ir.Expr
      case class Inner(value: Expr[Int]) derives ProductFoldState
      case class Outer(inner: Inner) derives ProductFoldState
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.dsl.ProductFoldState
      import flight4s.core.ir.Expr
      case class State(value: Expr[Int]) derives ProductFoldState
      kernel("readonlyProduct") {
        val state = gpuRange("i", literal(0), literal(4)).foldLeft("state", State(literal(0)))((s, _) => s)
        state.value := literal(1)
      }
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.ProductFoldState
      import flight4s.core.ir.Expr
      enum Choice derives ProductFoldState:
        case One(value: Expr[Int])
        case Two(value: Expr[Float])
    """).nonEmpty)

  test("generic case classes support explicit derivation with CUDA field evidence"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.dsl.ProductFoldState
      import flight4s.core.ir.Expr
      import flight4s.core.types.CudaType
      case class State[A](value: Expr[A])
      object State:
        given [A: CudaType]: ProductFoldState[State[A]] = ProductFoldState.derived
      kernel("genericProduct") {
        gpuRange("i", literal(0), literal(4)).foldLeft("state", State(literal(1.0f))) {
          (s, i) => State(s.value + convert.i32ToF32(i))
        }
        ()
      }
    """), Nil)
