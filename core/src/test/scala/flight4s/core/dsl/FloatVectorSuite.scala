package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class FloatVectorSuite extends FunSuite:
  test("Float2 and Float4 are typed CUDA values with component composition"):
    assertEquals(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.ir.Expr
      import flight4s.core.types.{Float2, Float4}
      kernel("vectors", params(input[Float4]("source"), output[Float4]("out"))) { p =>
        val v: Expr[Float4] = let("v", p._1(threadIdx.x).read)
        val pair: Expr[Float2] = float2(v.x, v.y)
        val rebuilt = float4(pair.x, pair.y, v.z, v.w)
        p._2(threadIdx.x) := rebuilt.map(_ * literal(2.0f)).zipWith(v)(_ + _)
      }
    """), Nil)

  test("vector sizes and alignments match CUDA native vector storage"):
    assertEquals((F32x2.sizeBytes, F32x2.alignmentBytes, F32x2.componentCount), (8, 8, 2))
    assertEquals((F32x4.sizeBytes, F32x4.alignmentBytes, F32x4.componentCount), (16, 16, 4))

  test("vector types reject unsupported operators, components, scalar ABI and mixed zip widths"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.*
      val v = literal(Float2(1f, 2f))
      v.z
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.*
      val v = literal(Float4(1f, 2f, 3f, 4f))
      v + v
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.*
      value[Float4]("v")
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.*
      literal(Float2(1f, 2f)).zipWith(literal(Float4(1f, 2f, 3f, 4f)))(_ + _)
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.types.*
      literal(Float4(1f, 2f, 3f, 4f)).map(_ => literal[Double](1.0))
    """).nonEmpty)

  test("raw vector IR rejects invalid arity component and native vector operators"):
    val span = SourceSpan("vectors.scala", 8, 3, 8, 30)
    val v = literal(Float2(1f, 2f))
    val cases: Vector[(Expr[?], ValidationCode)] = Vector(
      FloatVectorConstruct(Vector(literal(1f)), F32x2, span) -> ValidationCode.InvalidVectorArity,
      FloatVectorComponent(v, -1, F32x2, span) -> ValidationCode.InvalidVectorComponent,
      FloatVectorComponent(v, 2, F32x2, span) -> ValidationCode.InvalidVectorComponent,
      Binary(BinaryOperator.Add, v, v, F32x2, span) -> ValidationCode.UnsupportedVectorOperation,
      Compare(ComparisonOperator.Equal, v, v, F32x2, span) -> ValidationCode.UnsupportedVectorOperation
    )
    def definition[T](expr: Expr[T]) =
      given CudaType[T] = expr.valueType
      kernel("invalid") { let("bad", expr); () }
    for (expr, code) <- cases do
      val k = definition(expr)
      val errors = KernelValidator.validate(k).errors
      assert(errors.exists(error => error.code == code && error.span == span), errors.toString)
      assert(CudaCodegen.generate(k).isLeft)

  test("components retain operand effects uniformity validation and source locations"):
    val source = input[Float]("source")
    val v = float2(literal(1f), source(threadIdx.x).read)
    assertEquals(EffectAnalysis.expression(v.x), EffectAnalysis.expression(source(threadIdx.x).read))
    assertEquals(UniformityAnalysis.expression(v.x), Uniformity.Varying)
    val missing = kernel("missing") { let("x", v.x); () }
    assert(KernelValidator.validate(missing).errors.nonEmpty)
    assert(CudaCodegen.generate(missing).isLeft)
    val varying = kernel("varying", params(source)) { _ =>
      val vector = let("v", v)
      when(vector.x > literal(0f)) { barrier() }
    }
    assertEquals(KernelValidator.validate(varying).warnings.map(_.code), Vector(ValidationWarningCode.BarrierMayDiverge))
    assertNotEquals(v.span, SourceSpan.Unknown)
    assertNotEquals(v.x.span, SourceSpan.Unknown)

  test("component composition is expression-only and stages once per component"):
    var calls = 0
    val v = literal(Float4(1f, 2f, 3f, 4f))
    val mapped = v.map { x => calls += 1; x + literal(1f) }
    val zipped = v.zipWith(mapped) { (x, y) => calls += 1; x * y }
    assertEquals(calls, 8)
    assertEquals(zipped.valueType, F32x4)
    for useZip <- Vector(false, true) do
      val error = intercept[DslError] {
        kernel("effects") {
          if useZip then v.zipWith(v) { (x, y) => barrier(); x + y }
          else v.map { x => barrier(); x }
          ()
        }
      }
      assertEquals(error.code, DslErrorCode.StatementInsideExpression)

  test("vector lowering emits native aggregates and recursive normalization keeps effects"):
    val source = input[Float4]("source")
    val out = output[Float4]("out")
    val definition = kernel("vectors", params(source, out)) { _ =>
      val v = let("v", source(literal(2) + literal(3)).read)
      out(threadIdx.x) := v.map(_ * literal(2f)).zipWith(literal(Float4(-0f, 1f, 2f, 3f)))(_ + _)
    }
    val normalized = IrNormalizer.kernel(definition.ir)
    assertEquals(EffectAnalysis.block(normalized.body), EffectAnalysis.block(definition.body))
    val optimized = LocalCommonSubexpressionElimination.kernel(normalized)
    assertEquals(EffectAnalysis.block(optimized.body), EffectAnalysis.block(normalized.body))
    val generated = CudaCodegen.generate(definition).toOption.get.cudaSource
    assert(generated.contains("#include <vector_types.h>"), generated)
    assert(generated.contains("float4{"), generated)
    assert(generated.contains("source[5]"), generated)
    for field <- Vector("x", "y", "z", "w") do assert(generated.contains(s"(v).$field"), generated)

  test("vector headers are discovered in expression-only types"):
    val definition = kernel("scalarOutput", params(output[Float]("out"))) { out =>
      out._1(threadIdx.x) := literal(Float4(-0.0f, Float.PositiveInfinity, Float.NegativeInfinity, Float.NaN)).w
    }
    val generated = CudaCodegen.generate(definition).toOption.get.cudaSource
    assert(generated.contains("#include <vector_types.h>"), generated)
    assert(generated.contains("float4{"), generated)

  test("CUDA vector type names cannot be shadowed by module or kernel bindings"):
    for name <- Vector("float2", "float4") do
      val kernels = Vector(
        kernel(name) { () },
        kernel("parameter", params(input[Float4](name))) { _ => () },
        kernel("local") { local(name, literal(1)); let("v", literal(Float4(1f, 2f, 3f, 4f))); () },
        kernel("shared") { sharedArray[Float4](name, 1); () },
        kernel("array") { localArray[Float4](name, 1); () },
        kernel("loop") { gpuFor(name, literal(0), literal(1)) { _ => () } }
      )
      for definition <- kernels do assert(KernelValidator.validate(definition).errors.nonEmpty)
      assert(ModuleValidator.validate(module(constants = Vector(constantArray[Float4](name, 1)))).errors.nonEmpty)
