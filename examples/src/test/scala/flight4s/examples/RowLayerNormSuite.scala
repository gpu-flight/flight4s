package flight4s.examples

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.ir.*

class RowLayerNormSuite extends FunSuite:
  test("row layer normalization accepts explicit gain bias and epsilon with Float output"):
    assertEquals(typeCheckErrors("""
      import flight4s.examples.RowLayerNorm
      val result: Array[Float] = RowLayerNorm.run(
        Array(1.0f, 3.0f), gain = Array(1.0f, 2.0f), bias = Array(0.0f, 1.0f),
        rows = 1, columns = 2, epsilon = 1e-5)
    """), Nil)

  test("invalid shapes fail before native setup"):
    for (values, rows, columns) <- Vector(
      (Array.emptyFloatArray, -1, 3), (Array.emptyFloatArray, 1, 0),
      (Array.emptyFloatArray, 0, -1), (Array(1.0f), 1, 2),
      (Array.emptyFloatArray, 50000, 50000), (Array(1.0f), 0, 1)
    ) do intercept[IllegalArgumentException](
      RowLayerNorm.run(values, Array(1.0f), Array(0.0f), rows, columns, deviceOrdinal = -1))
    for (gain, bias) <- Vector((Array.emptyFloatArray, Array(0.0f)),
      (Array(1.0f), Array.emptyFloatArray), (Array(1.0f, 1.0f), Array(0.0f))) do
      intercept[IllegalArgumentException](RowLayerNorm.run(Array(1.0f), gain, bias, 1, 1, deviceOrdinal = -1))

  test("non-finite inputs and non-positive epsilon fail before native setup"):
    for invalid <- Vector(Float.NaN, Float.PositiveInfinity, Float.NegativeInfinity) do
      for (values, gain, bias) <- Vector(
        (Array(invalid), Array(1.0f), Array(0.0f)),
        (Array(1.0f), Array(invalid), Array(0.0f)),
        (Array(1.0f), Array(1.0f), Array(invalid))) do
        intercept[IllegalArgumentException](RowLayerNorm.run(values, gain, bias, 1, 1, deviceOrdinal = -1))
    for epsilon <- Vector(0.0, -0.0, -1.0, Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity) do
      intercept[IllegalArgumentException](RowLayerNorm.run(Array(1.0f), Array(1.0f), Array(0.0f),
        1, 1, epsilon, deviceOrdinal = -1))

  test("empty batches avoid CUDA but still validate affine parameters and epsilon"):
    assertEquals(RowLayerNorm.run(Array.emptyFloatArray, Array(1.0f), Array(0.0f),
      0, 1, deviceOrdinal = -1).toVector, Vector.empty)
    intercept[IllegalArgumentException](RowLayerNorm.run(Array.emptyFloatArray, Array(1.0f), Array(0.0f),
      0, 1, epsilon = 0.0, deviceOrdinal = -1))
    intercept[IllegalArgumentException](RowLayerNorm.run(Array.emptyFloatArray, Array.emptyFloatArray,
      Array(0.0f), 0, 1, deviceOrdinal = -1))
    for epsilon <- Vector(java.lang.Double.MIN_VALUE, 1e-5, Double.MaxValue) do
      RowLayerNorm.validateInput(Array(Float.MaxValue), Array(-Float.MaxValue), Array(Float.MaxValue), 1, 1, epsilon)

  test("the staged kernel uses a tuple fold then an output traversal with explicit precision"):
    assertEquals(KernelValidator.validate(RowLayerNorm.definition).errors, Vector.empty)
    val guard = RowLayerNorm.definition.body.statements.last.asInstanceOf[IfThen]
    val loops = guard.thenBlock.statements.collect { case loop: ForLoop => loop }
    assertEquals(loops.size, 2)
    assertEquals(loops.head.body.statements.size, 6)
    assert(loops.head.body.statements.take(3).forall(_.isInstanceOf[LocalDeclaration[?]]))
    assert(loops.head.body.statements.drop(3).forall(_.isInstanceOf[Store[?, ?]]))
    val source = RowLayerNorm.generated.cudaSource
    assert(source.contains("double moments_1"))
    assert(source.contains("double moments_2"))
    assert(source.contains("double denominator = ::sqrt("))
    assert(source.contains("::__double2float_rn("))
    assert(source.contains("static_cast<double>(gain[outputColumn])"))
    assert(source.contains("static_cast<double>(bias[outputColumn])"))
    assert(!source.contains("__syncthreads"))
    assertEquals(RowLayerNorm.generated.compilerOptions.additionalNvrtcOptions, Vector.empty)

  test("source inspection and CLI argument errors require no GPU"):
    val output = new java.io.ByteArrayOutputStream
    Console.withOut(output) { RowLayerNorm.main(Array("--cuda-source")) }
    assert(output.toString("UTF-8").contains(RowLayerNorm.generated.cudaSource))
    intercept[IllegalArgumentException](RowLayerNorm.main(Array("--invalid")))
