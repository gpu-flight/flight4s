package flight4s.core.ir

import scala.compiletime.testing.typeCheckErrors

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

class KernelSignatureSuite extends FunSuite:
  test("a descriptor tuple derives bindings and launch argument types"):
    val signature = params(
      in[Float]("left"),
      in[Float]("right"),
      out[Float]("result"),
      value[Int]("size")
    )

    val typedSignature: KernelSignature[
      (
          DeviceBuffer[Float],
          DeviceBuffer[Float],
          DeviceBuffer[Float],
          Int
      )
    ] = signature
    val bindings = signature.bindings
    val left: BufferParam[Float, ReadOnly] = bindings.head
    val right: BufferParam[Float, ReadOnly] = bindings.tail.head
    val result: BufferParam[Float, ReadWrite] = bindings.tail.tail.head
    val size: Expr[Int] = bindings.tail.tail.tail.head

    assertEquals(typedSignature.parameters.map(_.name), Vector("left", "right", "result", "size"))
    assertEquals(left.access, BufferAccess.ReadOnly)
    assertEquals(right.access, BufferAccess.ReadOnly)
    assertEquals(result.access, BufferAccess.ReadWrite)
    assertEquals(size.valueType, I32)

  test("one signature instance is retained by Kernel and KernelIR"):
    val signature = params(
      in[Float]("left"),
      in[Float]("right"),
      out[Float]("result"),
      value[Int]("size")
    )
    val vectorAdd: Kernel[
      (
          DeviceBuffer[Float],
          DeviceBuffer[Float],
          DeviceBuffer[Float],
          Int
      )
    ] =
      kernel("vectorAdd", signature) { bindings =>
        val left = bindings.head
        val right = bindings.tail.head
        val result = bindings.tail.tail.head
        val size = bindings.tail.tail.tail.head
        val index = blockIdx.x * blockDim.x + threadIdx.x

        when(index < size) {
          result(index) := left(index).read + right(index).read
        }
      }

    assert(vectorAdd.signature eq signature)
    assert(vectorAdd.ir.signature eq signature)
    assert(KernelValidator.validate(vectorAdd).isValid)

  test("Kernel.bind retains the statically checked argument tuple"):
    val signature = params(value[Int]("size"), value[Float]("scale"))
    val definition: Kernel[(Int, Float)] =
      kernel("scale", signature) { _ => () }
    val invocation = definition.bind((128, 0.5f))

    assertEquals(invocation.arguments, (128, 0.5f))
    assertEquals(invocation.kernel, definition)

  test("incorrect launch argument count fails to compile"):
    val errors = typeCheckErrors(
      """
        import flight4s.core.dsl.CudaDsl.*

        val signature = params(value[Int]("size"), value[Float]("scale"))
        val definition = kernel("scale", signature) { _ => () }
        val invalid = definition.bind(Tuple1(128))
      """
    )

    assert(errors.nonEmpty)

  test("incorrect launch argument types fail to compile"):
    val errors = typeCheckErrors(
      """
        import flight4s.core.dsl.CudaDsl.*

        val signature = params(value[Int]("size"), value[Float]("scale"))
        val definition = kernel("scale", signature) { _ => () }
        val invalid = definition.bind((128, true))
      """
    )

    assert(errors.nonEmpty)

  test("buffer element types remain part of the launch contract"):
    val errors = typeCheckErrors(
      """
        import flight4s.core.dsl.CudaDsl.*
        import flight4s.core.ir.DeviceBuffer

        val signature = params(in[Float]("source"))
        val definition = kernel("copy", signature) { _ => () }
        def wrongBuffer: DeviceBuffer[Double] = ???
        val invalid = definition.bind(Tuple1(wrongBuffer))
      """
    )

    assert(errors.nonEmpty)

  test("tuple input supports signatures beyond convenience overload arity"):
    val signature = params(
      (
        value[Int]("p1"),
        value[Int]("p2"),
        value[Int]("p3"),
        value[Int]("p4"),
        value[Int]("p5"),
        value[Int]("p6"),
        value[Int]("p7")
      )
    )
    val definition: Kernel[(Int, Int, Int, Int, Int, Int, Int)] =
      kernel("manyParameters", signature) { _ => () }
    val invocation = definition.bind((1, 2, 3, 4, 5, 6, 7))

    assertEquals(invocation.arguments.productArity, 7)

  test("scalar parameter expressions must belong to the kernel signature"):
    val declared = value[Int]("declared")
    val missing = value[Int]("missing")
    val result = out[Int]("result")
    val definition = Kernel(
      KernelIR(
        "missingScalar",
        params(declared, result),
        Block(
          Vector(
            Store(
              BufferElement[Int, ReadWrite]("result", literal(0), I32),
              missing
            )
          )
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnknownScalarParameter))

  test("inline tuple factories evaluate bindings once and retain the runtime signature contract"):
    var evaluations = 0
    def bindings() =
      evaluations += 1
      (input[Float]("data"), output[Float]("target"), value[Int]("count"))
    val signature = paramsTuple(bindings())
    assertEquals(evaluations, 1)
    val supplied = KernelSignature.fromTupleWithEvidence(signature.bindings)
    val typed: KernelSignature[(DeviceBuffer[Float], DeviceBuffer[Float], Int)] = signature
    assert(supplied.bindings eq signature.bindings)
    assertEquals(supplied.parameters, typed.parameters)
    assertEquals(supplied.abiDescriptors, typed.abiDescriptors)
