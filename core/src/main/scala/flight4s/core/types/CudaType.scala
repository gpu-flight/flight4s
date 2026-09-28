package flight4s.core.types

sealed trait CudaType[T]:
  def cudaName: String
  def sizeBytes: Int
  def alignmentBytes: Int
  def requiredHeaders: Set[String] = Set.empty

sealed trait AdditiveType[T] extends CudaType[T]
sealed trait MultiplicativeType[T] extends CudaType[T]
sealed trait DivisibleType[T] extends CudaType[T]
sealed trait RemainderType[T] extends CudaType[T]
sealed trait BitwiseType[T] extends CudaType[T]:
  private[core] def allBitsSet: T
sealed trait OrderedType[T] extends CudaType[T]
sealed trait EqualityComparableType[T] extends CudaType[T]
sealed trait FloatingMathType[T] extends CudaType[T]
sealed trait AtomicAddType[T] extends CudaType[T]
sealed trait AtomicValueType[T] extends CudaType[T]
sealed trait AtomicIntegralType[T] extends AtomicValueType[T]
sealed trait WarpShuffleType[T] extends CudaType[T]
sealed trait FloatVectorType[T] extends CudaType[T]:
  def componentCount: Int
  override val requiredHeaders: Set[String] = Set("vector_types.h")

sealed trait AccumulatorType[Input, Accumulator]:
  def inputType: CudaType[Input]
  def accumulatorType: CudaType[Accumulator]

final case class Float2(x: Float, y: Float)
final case class Float4(x: Float, y: Float, z: Float, w: Float)

final case class UInt private (bits: Int) extends AnyVal

object UInt:
  def fromBits(bits: Int): UInt = UInt(bits)

  extension (value: UInt)
    def toIntBits: Int = value.bits

final case class Float16 private (bits: Short) extends AnyVal

object Float16:
  def fromBits(bits: Short): Float16 = Float16(bits)

  extension (value: Float16)
    def toShortBits: Short = value.bits

final case class BFloat16 private (bits: Short) extends AnyVal

object BFloat16:
  def fromBits(bits: Short): BFloat16 = BFloat16(bits)

  extension (value: BFloat16)
    def toShortBits: Short = value.bits

final case class Float8E4M3 private (bits: Byte) extends AnyVal

object Float8E4M3:
  def fromBits(bits: Byte): Float8E4M3 = Float8E4M3(bits)

  extension (value: Float8E4M3)
    def toByteBits: Byte = value.bits

final case class Float8E5M2 private (bits: Byte) extends AnyVal

object Float8E5M2:
  def fromBits(bits: Byte): Float8E5M2 = Float8E5M2(bits)

  extension (value: Float8E5M2)
    def toByteBits: Byte = value.bits

case object Bool
    extends CudaType[Boolean],
      EqualityComparableType[Boolean]:
  override val cudaName: String = "bool"
  override val sizeBytes: Int = 1
  override val alignmentBytes: Int = 1

case object I32
    extends AdditiveType[Int],
      AtomicIntegralType[Int],
      AtomicAddType[Int],
      WarpShuffleType[Int],
      MultiplicativeType[Int],
      DivisibleType[Int],
      RemainderType[Int],
      BitwiseType[Int],
      OrderedType[Int],
      EqualityComparableType[Int]:
  override val cudaName: String = "int"
  override private[core] val allBitsSet: Int = -1
  override val sizeBytes: Int = 4
  override val alignmentBytes: Int = 4

case object U32
    extends AdditiveType[UInt],
      AtomicIntegralType[UInt],
      AtomicAddType[UInt],
      WarpShuffleType[UInt],
      MultiplicativeType[UInt],
      DivisibleType[UInt],
      RemainderType[UInt],
      BitwiseType[UInt],
      OrderedType[UInt],
      EqualityComparableType[UInt]:
  override val cudaName: String = "unsigned int"
  override private[core] val allBitsSet: UInt = UInt.fromBits(-1)
  override val sizeBytes: Int = 4
  override val alignmentBytes: Int = 4

case object F16
    extends AdditiveType[Float16],
      MultiplicativeType[Float16],
      DivisibleType[Float16],
      OrderedType[Float16],
      EqualityComparableType[Float16]:
  override val cudaName: String = "__half"
  override val sizeBytes: Int = 2
  override val alignmentBytes: Int = 2
  override val requiredHeaders: Set[String] = Set("cuda_fp16.h")

case object BF16
    extends AdditiveType[BFloat16],
      MultiplicativeType[BFloat16],
      DivisibleType[BFloat16],
      OrderedType[BFloat16],
      EqualityComparableType[BFloat16]:
  override val cudaName: String = "__nv_bfloat16"
  override val sizeBytes: Int = 2
  override val alignmentBytes: Int = 2
  override val requiredHeaders: Set[String] = Set("cuda_bf16.h")

case object F32
    extends AdditiveType[Float],
      AtomicValueType[Float],
      AtomicAddType[Float],
      WarpShuffleType[Float],
      FloatingMathType[Float],
      MultiplicativeType[Float],
      DivisibleType[Float],
      OrderedType[Float],
      EqualityComparableType[Float]:
  override val cudaName: String = "float"
  override val sizeBytes: Int = 4
  override val alignmentBytes: Int = 4

case object F64
    extends AdditiveType[Double],
      AtomicValueType[Double],
      WarpShuffleType[Double],
      FloatingMathType[Double],
      MultiplicativeType[Double],
      DivisibleType[Double],
      OrderedType[Double],
      EqualityComparableType[Double]:
  override val cudaName: String = "double"
  override val sizeBytes: Int = 8
  override val alignmentBytes: Int = 8

case object F32x2 extends FloatVectorType[Float2]:
  override val cudaName: String = "float2"
  override val componentCount: Int = 2
  override val sizeBytes: Int = 8
  override val alignmentBytes: Int = 8

case object F32x4 extends FloatVectorType[Float4]:
  override val cudaName: String = "float4"
  override val componentCount: Int = 4
  override val sizeBytes: Int = 16
  override val alignmentBytes: Int = 16

object FloatVectorType:
  given float2Type: FloatVectorType[Float2] = F32x2
  given float4Type: FloatVectorType[Float4] = F32x4

case object FP8E4M3 extends CudaType[Float8E4M3]:
  override val cudaName: String = "__nv_fp8_e4m3"
  override val sizeBytes: Int = 1
  override val alignmentBytes: Int = 1
  override val requiredHeaders: Set[String] = Set("cuda_fp8.h")

case object FP8E5M2 extends CudaType[Float8E5M2]:
  override val cudaName: String = "__nv_fp8_e5m2"
  override val sizeBytes: Int = 1
  override val alignmentBytes: Int = 1
  override val requiredHeaders: Set[String] = Set("cuda_fp8.h")

object CudaType:
  given float2Type: CudaType[Float2] = F32x2
  given float4Type: CudaType[Float4] = F32x4
  given boolType: CudaType[Boolean] = Bool
  given intType: CudaType[Int] = I32
  given uintType: CudaType[UInt] = U32
  given float16Type: CudaType[Float16] = F16
  given bfloat16Type: CudaType[BFloat16] = BF16
  given floatType: CudaType[Float] = F32
  given doubleType: CudaType[Double] = F64
  given float8E4M3Type: CudaType[Float8E4M3] = FP8E4M3
  given float8E5M2Type: CudaType[Float8E5M2] = FP8E5M2

object AdditiveType:
  given intType: AdditiveType[Int] = I32
  given uintType: AdditiveType[UInt] = U32
  given float16Type: AdditiveType[Float16] = F16
  given bfloat16Type: AdditiveType[BFloat16] = BF16
  given floatType: AdditiveType[Float] = F32
  given doubleType: AdditiveType[Double] = F64

object MultiplicativeType:
  given intType: MultiplicativeType[Int] = I32
  given uintType: MultiplicativeType[UInt] = U32
  given float16Type: MultiplicativeType[Float16] = F16
  given bfloat16Type: MultiplicativeType[BFloat16] = BF16
  given floatType: MultiplicativeType[Float] = F32
  given doubleType: MultiplicativeType[Double] = F64

object DivisibleType:
  given intType: DivisibleType[Int] = I32
  given uintType: DivisibleType[UInt] = U32
  given float16Type: DivisibleType[Float16] = F16
  given bfloat16Type: DivisibleType[BFloat16] = BF16
  given floatType: DivisibleType[Float] = F32
  given doubleType: DivisibleType[Double] = F64

object RemainderType:
  given intType: RemainderType[Int] = I32
  given uintType: RemainderType[UInt] = U32

object BitwiseType:
  given intType: BitwiseType[Int] = I32
  given uintType: BitwiseType[UInt] = U32

object OrderedType:
  given intType: OrderedType[Int] = I32
  given uintType: OrderedType[UInt] = U32
  given float16Type: OrderedType[Float16] = F16
  given bfloat16Type: OrderedType[BFloat16] = BF16
  given floatType: OrderedType[Float] = F32
  given doubleType: OrderedType[Double] = F64

object EqualityComparableType:
  given boolType: EqualityComparableType[Boolean] = Bool
  given intType: EqualityComparableType[Int] = I32
  given uintType: EqualityComparableType[UInt] = U32
  given float16Type: EqualityComparableType[Float16] = F16
  given bfloat16Type: EqualityComparableType[BFloat16] = BF16
  given floatType: EqualityComparableType[Float] = F32
  given doubleType: EqualityComparableType[Double] = F64

object FloatingMathType:
  given floatType: FloatingMathType[Float] = F32
  given doubleType: FloatingMathType[Double] = F64

object AtomicAddType:
  given intType: AtomicAddType[Int] = I32
  given uintType: AtomicAddType[UInt] = U32
  given floatType: AtomicAddType[Float] = F32

object AtomicValueType:
  given intType: AtomicValueType[Int] = I32
  given uintType: AtomicValueType[UInt] = U32
  given floatType: AtomicValueType[Float] = F32
  given doubleType: AtomicValueType[Double] = F64

object AtomicIntegralType:
  given intType: AtomicIntegralType[Int] = I32
  given uintType: AtomicIntegralType[UInt] = U32

object WarpShuffleType:
  given intType: WarpShuffleType[Int] = I32
  given uintType: WarpShuffleType[UInt] = U32
  given floatType: WarpShuffleType[Float] = F32
  given doubleType: WarpShuffleType[Double] = F64

object AccumulatorType:
  given intAccumulator: AccumulatorType[Int, Int] with
    override val inputType: CudaType[Int] = I32
    override val accumulatorType: CudaType[Int] = I32

  given uintAccumulator: AccumulatorType[UInt, UInt] with
    override val inputType: CudaType[UInt] = U32
    override val accumulatorType: CudaType[UInt] = U32

  given float16Accumulator: AccumulatorType[Float16, Float] with
    override val inputType: CudaType[Float16] = F16
    override val accumulatorType: CudaType[Float] = F32

  given bfloat16Accumulator: AccumulatorType[BFloat16, Float] with
    override val inputType: CudaType[BFloat16] = BF16
    override val accumulatorType: CudaType[Float] = F32

  given floatAccumulator: AccumulatorType[Float, Float] with
    override val inputType: CudaType[Float] = F32
    override val accumulatorType: CudaType[Float] = F32

  given doubleAccumulator: AccumulatorType[Double, Double] with
    override val inputType: CudaType[Double] = F64
    override val accumulatorType: CudaType[Double] = F64

  given float8E4M3Accumulator: AccumulatorType[Float8E4M3, Float] with
    override val inputType: CudaType[Float8E4M3] = FP8E4M3
    override val accumulatorType: CudaType[Float] = F32

  given float8E5M2Accumulator: AccumulatorType[Float8E5M2, Float] with
    override val inputType: CudaType[Float8E5M2] = FP8E5M2
    override val accumulatorType: CudaType[Float] = F32
