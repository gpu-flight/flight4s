package flight4s.examples

import munit.FunSuite
import flight4s.core.ir.KernelValidator

class RowSoftmaxSuite extends FunSuite:
  test("input validation rejects invalid shapes and non-finite logits before touching CUDA"):
    intercept[IllegalArgumentException](RowSoftmax.validateInput(Array.emptyFloatArray, -1, 3))
    intercept[IllegalArgumentException](RowSoftmax.validateInput(Array.emptyFloatArray, 1, 0))
    intercept[IllegalArgumentException](RowSoftmax.validateInput(Array(1.0f), 1, 2))
    intercept[IllegalArgumentException](RowSoftmax.validateInput(Array.emptyFloatArray, 50000, 50000))
    for value <- Vector(Float.NaN, Float.PositiveInfinity, Float.NegativeInfinity) do
      intercept[IllegalArgumentException](RowSoftmax.validateInput(Array(value), 1, 1))

  test("launch coverage rounds up without overflowing the host row count"):
    assertEquals(RowSoftmax.blockCount(1), 1)
    assertEquals(RowSoftmax.blockCount(128), 1)
    assertEquals(RowSoftmax.blockCount(129), 2)
    assertEquals(RowSoftmax.blockCount(Int.MaxValue), 16777216)

  test("empty batches return without native setup and valid finite shapes are accepted"):
    RowSoftmax.validateInput(Array(0.0f, -1.0f, Float.MaxValue), 1, 3)
    assertEquals(RowSoftmax.run(Array.emptyFloatArray, 0, 3).toVector, Vector.empty)

  test("the complete example kernel validates and generates inspectable CUDA without a GPU"):
    assert(KernelValidator.validate(RowSoftmax.definition).isValid)
    assert(RowSoftmax.generated.cudaSource.contains("::expf("))
    assert(RowSoftmax.generated.cudaSource.contains("float maximum"))
    assert(RowSoftmax.generated.cudaSource.contains("float denominator"))
    assertEquals(RowSoftmax.generated.compilerOptions.additionalNvrtcOptions, Vector.empty)
