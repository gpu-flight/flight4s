package flight4s.runtime.cuda

import java.nio.{ByteBuffer, ByteOrder}

import flight4s.core.types.*

sealed trait CudaHostCodec[T]:
  def cudaType: CudaType[T]

  private[cuda] def encode(values: Array[T]): ByteBuffer
  private[cuda] def encodeInto(
      values: Array[T],
      destination: ByteBuffer
  ): Unit
  private[cuda] def decode(bytes: ByteBuffer, elementCount: Int): Array[T]

object CudaHostCodec:
  private final class FixedWidthCodec[T](
      override val cudaType: CudaType[T],
      allocate: Int => Array[T],
      write: (ByteBuffer, T) => Unit,
      read: ByteBuffer => T
  ) extends CudaHostCodec[T]:
    override private[cuda] def encode(values: Array[T]): ByteBuffer =
      val sizeBytes =
        Math.multiplyExact(values.length, cudaType.sizeBytes)
      val bytes = ByteBuffer
        .allocateDirect(sizeBytes)
        .order(ByteOrder.nativeOrder())
      encodeInto(values, bytes)
      bytes

    override private[cuda] def encodeInto(
        values: Array[T],
        destination: ByteBuffer
    ): Unit =
      val sizeBytes =
        Math.multiplyExact(values.length, cudaType.sizeBytes)
      require(destination.isDirect, "host codec destination must be direct")
      require(
        destination.capacity() == sizeBytes,
        s"host codec destination capacity ${destination.capacity()} " +
          s"does not match encoded size $sizeBytes"
      )
      val output = destination
        .duplicate()
        .order(ByteOrder.nativeOrder())
      output.clear()
      var index = 0
      while index < values.length do
        write(output, values(index))
        index += 1

    override private[cuda] def decode(
        bytes: ByteBuffer,
        elementCount: Int
    ): Array[T] =
      val input = bytes.duplicate().order(ByteOrder.nativeOrder())
      val values = allocate(elementCount)
      var index = 0
      while index < elementCount do
        values(index) = read(input)
        index += 1
      values

  given booleanCodec: CudaHostCodec[Boolean] =
    FixedWidthCodec(
      Bool,
      new Array[Boolean](_),
      (bytes, value) => bytes.put(if value then 1.toByte else 0.toByte),
      bytes => bytes.get() != 0
    )

  given intCodec: CudaHostCodec[Int] =
    FixedWidthCodec(
      I32,
      new Array[Int](_),
      (bytes, value) => bytes.putInt(value),
      _.getInt()
    )

  given uintCodec: CudaHostCodec[UInt] =
    FixedWidthCodec(
      U32,
      new Array[UInt](_),
      (bytes, value) => bytes.putInt(value.toIntBits),
      bytes => UInt.fromBits(bytes.getInt())
    )

  given float16Codec: CudaHostCodec[Float16] =
    FixedWidthCodec(
      F16,
      new Array[Float16](_),
      (bytes, value) => bytes.putShort(value.toShortBits),
      bytes => Float16.fromBits(bytes.getShort())
    )

  given bfloat16Codec: CudaHostCodec[BFloat16] =
    FixedWidthCodec(
      BF16,
      new Array[BFloat16](_),
      (bytes, value) => bytes.putShort(value.toShortBits),
      bytes => BFloat16.fromBits(bytes.getShort())
    )

  given floatCodec: CudaHostCodec[Float] =
    FixedWidthCodec(
      F32,
      new Array[Float](_),
      (bytes, value) => bytes.putFloat(value),
      _.getFloat()
    )

  given doubleCodec: CudaHostCodec[Double] =
    FixedWidthCodec(
      F64,
      new Array[Double](_),
      (bytes, value) => bytes.putDouble(value),
      _.getDouble()
    )

  given float2Codec: CudaHostCodec[Float2] =
    FixedWidthCodec(F32x2, new Array[Float2](_),
      (bytes, value) => { bytes.putFloat(value.x); bytes.putFloat(value.y); () },
      bytes => Float2(bytes.getFloat(), bytes.getFloat()))

  given float4Codec: CudaHostCodec[Float4] =
    FixedWidthCodec(F32x4, new Array[Float4](_),
      (bytes, value) => { bytes.putFloat(value.x); bytes.putFloat(value.y); bytes.putFloat(value.z); bytes.putFloat(value.w); () },
      bytes => Float4(bytes.getFloat(), bytes.getFloat(), bytes.getFloat(), bytes.getFloat()))

  given float8E4M3Codec: CudaHostCodec[Float8E4M3] =
    FixedWidthCodec(
      FP8E4M3,
      new Array[Float8E4M3](_),
      (bytes, value) => bytes.put(value.toByteBits),
      bytes => Float8E4M3.fromBits(bytes.get())
    )

  given float8E5M2Codec: CudaHostCodec[Float8E5M2] =
    FixedWidthCodec(
      FP8E5M2,
      new Array[Float8E5M2](_),
      (bytes, value) => bytes.put(value.toByteBits),
      bytes => Float8E5M2.fromBits(bytes.get())
    )
