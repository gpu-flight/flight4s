package flight4s.core.ir

import java.nio.{ByteBuffer, ByteOrder}

import munit.FunSuite

import flight4s.core.abi.*
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

class KernelAbiSuite extends FunSuite:
  test("kernel signatures retain ordered native ABI descriptors"):
    val signature = params(
      in[Float]("input"),
      value[Int]("count"),
      value[UInt]("mask"),
      value[Float]("scale"),
      value[Double]("epsilon")
    )

    assertEquals(
      signature.abiDescriptors,
      Vector(
        KernelArgumentAbi("input", CudaAbiType.DevicePointer64),
        KernelArgumentAbi("count", CudaAbiType.SignedInt32),
        KernelArgumentAbi("mask", CudaAbiType.UnsignedInt32),
        KernelArgumentAbi("scale", CudaAbiType.Float32),
        KernelArgumentAbi("epsilon", CudaAbiType.Float64)
      )
    )
    assertEquals(
      signature.abiDescriptors.map(descriptor =>
        (
          descriptor.abiType.sizeBytes,
          descriptor.abiType.alignmentBytes
        )
      ),
      Vector((8, 8), (4, 4), (4, 4), (4, 4), (8, 8))
    )

  test("typed invocation packs pointers and scalars at exact ABI sizes"):
    val signature = params(
      in[Float]("input"),
      value[Int]("count"),
      value[UInt]("mask"),
      value[Float]("scale"),
      value[Double]("epsilon")
    )
    val definition = kernel("packed", signature) { _ => () }
    val input = TestDeviceBuffer[Float](0x1122334455667788L)
    val packed = definition
      .bind(
        (
          input,
          -17,
          UInt.fromBits(0xfedcba98),
          1.25f,
          0.0001d
        )
      )
      .packedArguments

    assertEquals(packed.descriptors, signature.abiDescriptors)
    assertEquals(packed.arguments.map(_.bytes.size), Vector(8, 4, 4, 4, 8))
    assertEquals(readLong(packed.arguments(0)), 0x1122334455667788L)
    assertEquals(readInt(packed.arguments(1)), -17)
    assertEquals(readInt(packed.arguments(2)), 0xfedcba98)
    assertEquals(readFloat(packed.arguments(3)), 1.25f)
    assertEquals(readDouble(packed.arguments(4)), 0.0001d)

  test("low-precision arguments preserve their declared raw bits"):
    val signature = params(
      value[Float16]("f16"),
      value[BFloat16]("bf16"),
      value[Float8E4M3]("e4m3"),
      value[Float8E5M2]("e5m2"),
      value[Boolean]("enabled")
    )
    val definition = kernel("lowPrecision", signature) { _ => () }
    val packed = definition
      .bind(
        (
          Float16.fromBits(0x3555.toShort),
          BFloat16.fromBits(0x3f80.toShort),
          Float8E4M3.fromBits(0x2a.toByte),
          Float8E5M2.fromBits(0x3b.toByte),
          true
        )
      )
      .packedArguments

    assertEquals(
      packed.descriptors.map(_.abiType),
      Vector(
        CudaAbiType.Float16,
        CudaAbiType.BFloat16,
        CudaAbiType.Float8E4M3,
        CudaAbiType.Float8E5M2,
        CudaAbiType.Bool8
      )
    )
    assertEquals(readShort(packed.arguments(0)), 0x3555.toShort)
    assertEquals(readShort(packed.arguments(1)), 0x3f80.toShort)
    assertEquals(packed.arguments(2).bytes, Vector(0x2a.toByte))
    assertEquals(packed.arguments(3).bytes, Vector(0x3b.toByte))
    assertEquals(packed.arguments(4).bytes, Vector(1.toByte))

  test("generic tuple and positional signatures produce identical packed native storage"):
    val bindings = (input[Float]("data"), output[Float]("target"), value[Int]("count"),
      value[Float]("factor"), value[Boolean]("enabled"), value[Double]("cutoff"))
    val positional = params(bindings._1, bindings._2, bindings._3, bindings._4, bindings._5, bindings._6)
    val generic = params(bindings)
    val explicit = KernelSignature.fromTupleWithEvidence(bindings)
    val arguments = (TestDeviceBuffer[Float](0x1122334455667788L), TestDeviceBuffer[Float](0x8877665544332211L),
      -17, 1.25f, true, 0.125)
    val before = kernel("packedTuple", positional)(_ => ()).bind(arguments)
    for signature <- Vector(generic, explicit) do
      val after = kernel("packedTuple", signature)(_ => ()).bind(arguments)
      assertEquals(signature.parameters, positional.parameters)
      assertEquals(signature.abiDescriptors, positional.abiDescriptors)
      assertEquals(after.packedArguments, before.packedArguments)
      assertEquals(after.nativeArguments.slots, before.nativeArguments.slots)
      assertEquals(after.nativeArguments.descriptorCodes, before.nativeArguments.descriptorCodes)
      def bytes(storage: NativeArgumentStorage): Vector[Byte] =
        val view = storage.bufferForLaunch
        Vector.tabulate(storage.sizeBytes)(view.get)
      assertEquals(bytes(after.nativeArguments), bytes(before.nativeArguments))

  test("seven parameter tuple packs ordered pointer scalar and padded Boolean Double slots"):
    val signature = params((input[Float]("data"), output[Float]("target"), value[Int]("count"),
      value[Float]("factor"), value[Float]("bias"), value[Boolean]("enabled"), value[Double]("cutoff")))
    val invocation = kernel("packedSeven", signature)(_ => ()).bind((
      TestDeviceBuffer[Float](0x1122334455667788L), TestDeviceBuffer[Float](0x8877665544332211L),
      -17, 1.25f, -0.5f, true, 0.125))
    val packed = invocation.packedArguments
    assertEquals(packed.descriptors.map(_.name), Vector("data", "target", "count", "factor", "bias", "enabled", "cutoff"))
    assertEquals(packed.descriptors.map(_.abiType), Vector(CudaAbiType.DevicePointer64, CudaAbiType.DevicePointer64,
      CudaAbiType.SignedInt32, CudaAbiType.Float32, CudaAbiType.Float32, CudaAbiType.Bool8, CudaAbiType.Float64))
    val expected = ByteBuffer.allocate(40).order(ByteOrder.nativeOrder())
    expected.putLong(0, 0x1122334455667788L)
    expected.putLong(8, 0x8877665544332211L)
    expected.putInt(16, -17)
    expected.putFloat(20, 1.25f)
    expected.putFloat(24, -0.5f)
    expected.put(28, 1.toByte)
    expected.putDouble(32, 0.125)
    val native = invocation.nativeArguments
    assertEquals(native.offsets, Vector(0, 8, 16, 20, 24, 28, 32))
    assertEquals(native.sizeBytes, 40)
    val actual = native.bufferForLaunch
    assertEquals(Vector.tabulate(40)(actual.get), expected.array().toVector)

  private final case class TestDeviceBuffer[T](rawAddress: Long)
      extends DeviceBuffer[T]:
    override private[flight4s] val deviceAddress: DeviceAddress =
      DeviceAddress.fromRaw(rawAddress)

  private def nativeBuffer(argument: PackedKernelArgument): ByteBuffer =
    ByteBuffer
      .wrap(argument.bytes.toArray)
      .order(ByteOrder.nativeOrder())

  private def readShort(argument: PackedKernelArgument): Short =
    nativeBuffer(argument).getShort()

  private def readInt(argument: PackedKernelArgument): Int =
    nativeBuffer(argument).getInt()

  private def readLong(argument: PackedKernelArgument): Long =
    nativeBuffer(argument).getLong()

  private def readFloat(argument: PackedKernelArgument): Float =
    nativeBuffer(argument).getFloat()

  private def readDouble(argument: PackedKernelArgument): Double =
    nativeBuffer(argument).getDouble()
