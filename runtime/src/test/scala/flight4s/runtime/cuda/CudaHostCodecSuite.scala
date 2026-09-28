package flight4s.runtime.cuda

import java.nio.{ByteBuffer, ByteOrder}

import munit.FunSuite

import flight4s.core.types.*

class CudaHostCodecSuite extends FunSuite:
  test("host codecs round-trip every supported CUDA scalar type"):
    assertRoundTrip(Array(false, true, true, false))
    assertRoundTrip(Array(Int.MinValue, -1, 0, Int.MaxValue))
    assertRoundTrip(
      Array(
        UInt.fromBits(Int.MinValue),
        UInt.fromBits(0),
        UInt.fromBits(Int.MaxValue)
      )
    )
    assertRoundTrip(
      Array(
        Float16.fromBits(Short.MinValue),
        Float16.fromBits(0),
        Float16.fromBits(Short.MaxValue)
      )
    )
    assertRoundTrip(
      Array(
        BFloat16.fromBits(Short.MinValue),
        BFloat16.fromBits(0),
        BFloat16.fromBits(Short.MaxValue)
      )
    )
    assertRoundTrip(
      Array(
        Float.NegativeInfinity,
        -0.0f,
        1.25f,
        Float.PositiveInfinity
      )
    )
    assertRoundTrip(
      Array(
        Double.NegativeInfinity,
        -0.0d,
        1.25d,
        Double.PositiveInfinity
      )
    )
    assertRoundTrip(
      Array(
        Float8E4M3.fromBits(Byte.MinValue),
        Float8E4M3.fromBits(0),
        Float8E4M3.fromBits(Byte.MaxValue)
      )
    )
    assertRoundTrip(
      Array(
        Float8E5M2.fromBits(Byte.MinValue),
        Float8E5M2.fromBits(0),
        Float8E5M2.fromBits(Byte.MaxValue)
      )
    )

  test("encoded host storage is an exact native-order direct view"):
    val codec = summon[CudaHostCodec[Int]]
    val bytes = codec.encode(Array(0x01020304, 0x11223344))

    assert(bytes.isDirect)
    assertEquals(bytes.position(), 0)
    assertEquals(bytes.limit(), 8)
    assertEquals(bytes.capacity(), 8)
    assertEquals(bytes.order(), ByteOrder.nativeOrder())
    assertEquals(
      bytes.duplicate().order(ByteOrder.nativeOrder()).getInt(),
      0x01020304
    )

  test("host codecs can reuse exact direct destination storage"):
    val codec = summon[CudaHostCodec[Int]]
    val destination = ByteBuffer
      .allocateDirect(8)
      .order(ByteOrder.nativeOrder())

    codec.encodeInto(Array(11, 22), destination)

    assertEquals(destination.position(), 0)
    assertEquals(destination.limit(), 8)
    val values = destination.duplicate().order(ByteOrder.nativeOrder())
    assertEquals(values.getInt(), 11)
    assertEquals(values.getInt(), 22)

  test("vector codecs preserve raw component bits and native-order packed layout"):
    val bits = Array(0x80000000, 0x7fc01234, 0x7f800000, 0xff800000, 0x00000001, 0x3f800000, 0, 0xbf800000)
    val floats = bits.map(java.lang.Float.intBitsToFloat)
    val pairs = floats.grouped(2).map(x => Float2(x(0), x(1))).toArray
    val quads = floats.grouped(4).map(x => Float4(x(0), x(1), x(2), x(3))).toArray
    def check[T](values: Array[T], flatten: Array[T] => Array[Float])(using codec: CudaHostCodec[T]): Unit =
      val bytes = codec.encode(values)
      assert(bytes.isDirect)
      assertEquals(bytes.capacity(), 32)
      assertEquals(bytes.order(), ByteOrder.nativeOrder())
      for i <- bits.indices do assertEquals(bytes.getInt(i * 4), bits(i))
      val decoded = flatten(codec.decode(bytes, values.length)).map(java.lang.Float.floatToRawIntBits)
      assertEquals(decoded.toVector, bits.toVector)
      val reused = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
      codec.encodeInto(values, reused)
      for i <- bits.indices do assertEquals(reused.getInt(i * 4), bits(i))
      assertEquals(codec.decode(codec.encode(values.take(0)), 0).length, 0)
    check(pairs, (xs: Array[Float2]) => xs.flatMap(x => Array(x.x, x.y)))
    check(quads, (xs: Array[Float4]) => xs.flatMap(x => Array(x.x, x.y, x.z, x.w)))

  private def assertRoundTrip[T](
      values: Array[T]
  )(using codec: CudaHostCodec[T]): Unit =
    val bytes = codec.encode(values)
    val decoded = codec.decode(bytes, values.length)

    assert(
      decoded.sameElements(values),
      s"${codec.cudaType.cudaName} host values did not round-trip"
    )
