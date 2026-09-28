package flight4s.core.ir

import flight4s.core.abi.ScalarAbi
import flight4s.core.types.{AccumulatorType, AdditiveType, AtomicAddType, BitwiseType, CudaType, FloatingMathType, I32, UInt, WarpShuffleType}

enum BinaryOperator(val cudaToken: String):
  case Add extends BinaryOperator("+")
  case Subtract extends BinaryOperator("-")
  case Multiply extends BinaryOperator("*")
  case Divide extends BinaryOperator("/")
  case Remainder extends BinaryOperator("%")
  case BitAnd extends BinaryOperator("&")
  case BitOr extends BinaryOperator("|")
  case BitXor extends BinaryOperator("^")

enum ComparisonOperator(val cudaToken: String):
  case LessThan extends ComparisonOperator("<")
  case LessThanOrEqual extends ComparisonOperator("<=")
  case GreaterThan extends ComparisonOperator(">")
  case GreaterThanOrEqual extends ComparisonOperator(">=")
  case Equal extends ComparisonOperator("==")
  case NotEqual extends ComparisonOperator("!=")

sealed trait Expr[T]:
  def valueType: CudaType[T]
  def span: SourceSpan

final case class Literal[T](
    value: T,
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]

final case class Binary[T](
    operator: BinaryOperator,
    left: Expr[T],
    right: Expr[T],
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]

enum UnsignedShiftOperator(val cudaToken: String):
  case Left extends UnsignedShiftOperator("<<")
  case Right extends UnsignedShiftOperator(">>")

final case class UnsignedShift(
    operator: UnsignedShiftOperator,
    value: Expr[UInt],
    distance: Expr[Int],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[UInt]:
  override val valueType: CudaType[UInt] = flight4s.core.types.U32

enum SignedShiftOperator:
  case Left, ArithmeticRight, LogicalRight

final case class SignedShift(
    operator: SignedShiftOperator,
    value: Expr[Int],
    distance: Expr[Int],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[Int]:
  override val valueType: CudaType[Int] = I32

final case class PopulationCount[T](
    value: Expr[T],
    wordType: BitwiseType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[Int]:
  override val valueType: CudaType[Int] = I32

final case class Compare[T](
    operator: ComparisonOperator,
    left: Expr[T],
    right: Expr[T],
    operandType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[Boolean]:
  override val valueType: CudaType[Boolean] = flight4s.core.types.Bool

final case class Conditional[T](
    condition: Expr[Boolean],
    whenTrue: Expr[T],
    whenFalse: Expr[T],
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]

enum UnaryMathOperator(val cudaName: String):
  case Exp extends UnaryMathOperator("exp")
  case Log extends UnaryMathOperator("log")
  case Sqrt extends UnaryMathOperator("sqrt")
  case Rsqrt extends UnaryMathOperator("rsqrt")
  case Tanh extends UnaryMathOperator("tanh")

final case class UnaryMath[T](
    operator: UnaryMathOperator,
    value: Expr[T],
    mathType: FloatingMathType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]:
  override def valueType: CudaType[T] = mathType

final case class Intrinsic[T](
    name: String,
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]

enum RoundingMode:
  case NearestEven
  case TowardZero
  case TowardPositive
  case TowardNegative

enum SaturationMode:
  case NoSaturation
  case SaturateFinite

final case class Convert[From, To](
    value: Expr[From],
    valueType: CudaType[To],
    rounding: RoundingMode,
    saturation: SaturationMode,
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[To]

final case class ToAccumulator[From, To](
    value: Expr[From],
    rule: AccumulatorType[From, To],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[To]:
  override def valueType: CudaType[To] = rule.accumulatorType

enum ReductionPolicy:
  case Strict
  case Deterministic
  case Fast

final case class ReductionIndex(
    name: String,
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[Int]:
  override val valueType: CudaType[Int] = I32

final case class LoopIndex(
    name: String,
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[Int]:
  override val valueType: CudaType[Int] = I32

final case class ReduceSum[Input, Accumulator](
    index: ReductionIndex,
    from: Expr[Int],
    until: Expr[Int],
    initial: Expr[Accumulator],
    value: Expr[Input],
    rule: AccumulatorType[Input, Accumulator],
    addition: AdditiveType[Accumulator],
    policy: ReductionPolicy,
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[Accumulator]:
  override def valueType: CudaType[Accumulator] = rule.accumulatorType

sealed trait AddressSpace
sealed trait Global extends AddressSpace
sealed trait Shared extends AddressSpace
sealed trait Local extends AddressSpace
sealed trait Constant extends AddressSpace

sealed trait AccessMode
sealed trait ReadOnly extends AccessMode
sealed trait ReadWrite extends AccessMode

enum BufferAccess:
  case ReadOnly
  case ReadWrite

sealed trait AccessModeWitness[Mode <: AccessMode]:
  def access: BufferAccess

object AccessModeWitness:
  given readOnlyWitness: AccessModeWitness[ReadOnly] with
    override val access: BufferAccess = BufferAccess.ReadOnly

  given readWriteWitness: AccessModeWitness[ReadWrite] with
    override val access: BufferAccess = BufferAccess.ReadWrite

sealed trait Place[T, Space <: AddressSpace, Mode <: AccessMode]:
  def valueType: CudaType[T]
  def span: SourceSpan

final case class BufferElement[T, Mode <: AccessMode](
    bufferName: String,
    index: Expr[Int],
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Place[T, Global, Mode]

final case class ConstantElement[T](
    arrayName: String,
    index: Expr[Int],
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Place[T, Constant, ReadOnly]

final case class SharedElement[T](
    arrayName: String,
    indices: Vector[Expr[Int]],
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Place[T, Shared, ReadWrite]

final case class LocalVariable[T](
    name: String,
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Place[T, Local, ReadWrite]

final case class LocalArrayElement[T](
    arrayName: String,
    index: Expr[Int],
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Place[T, Local, ReadWrite]

final case class Load[
    T,
    Space <: AddressSpace,
    Mode <: AccessMode
](
    from: Place[T, Space, Mode],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]:
  override def valueType: CudaType[T] = from.valueType

sealed trait Stmt:
  def span: SourceSpan

sealed trait ScopedDeclaration extends Stmt
sealed trait ExecutableStmt extends Stmt

final case class LocalDeclaration[T](
    local: LocalVariable[T],
    initial: Expr[T],
    span: SourceSpan = SourceSpan.Unknown
) extends ScopedDeclaration

final case class LocalArrayDeclaration[T](
    array: LocalArray[T],
    span: SourceSpan = SourceSpan.Unknown
) extends ScopedDeclaration

final case class WarpShuffle[T, S](
    local: LocalVariable[T],
    mask: Expr[UInt],
    value: Expr[T],
    operator: WarpShuffleOperator[S],
    selector: Expr[S],
    width: Int,
    shuffleType: WarpShuffleType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends ScopedDeclaration

final case class WarpVote[T](
    local: LocalVariable[T],
    operator: WarpVoteOperator[T],
    mask: Expr[UInt],
    predicate: Expr[Boolean],
    span: SourceSpan = SourceSpan.Unknown
) extends ScopedDeclaration

final case class Store[T, Space <: AddressSpace](
    to: Place[T, Space, ReadWrite],
    value: Expr[T],
    span: SourceSpan = SourceSpan.Unknown
) extends ExecutableStmt

final case class AtomicAdd[T, Space <: AddressSpace](
    target: Place[T, Space, ReadWrite],
    value: Expr[T],
    addition: AtomicAddType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends ExecutableStmt

final case class AtomicFetchAdd[T, Space <: AddressSpace](
    local: LocalVariable[T],
    target: Place[T, Space, ReadWrite],
    value: Expr[T],
    addition: AtomicAddType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends ScopedDeclaration

final case class Accumulate[T](
    target: LocalVariable[T],
    value: Expr[T],
    addition: AdditiveType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends ExecutableStmt

final case class IfThen(
    condition: Expr[Boolean],
    thenBlock: Block,
    elseBlock: Option[Block] = None,
    span: SourceSpan = SourceSpan.Unknown
) extends ExecutableStmt

final case class ScopedBlock(
    body: Block,
    span: SourceSpan = SourceSpan.Unknown
) extends ExecutableStmt

final case class ForLoop(
    index: LoopIndex,
    from: Expr[Int],
    until: Expr[Int],
    body: Block,
    span: SourceSpan = SourceSpan.Unknown
) extends ExecutableStmt

final case class Barrier(
    span: SourceSpan = SourceSpan.Unknown
) extends ExecutableStmt

final case class WarpBarrier(
    mask: Expr[UInt],
    span: SourceSpan = SourceSpan.Unknown
) extends ExecutableStmt

final case class Block(statements: Vector[Stmt])

sealed trait KernelParam:
  type Value

  def name: String
  def valueType: CudaType[Value]

final case class ScalarParam[T](
    name: String,
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
)(using val scalarAbi: ScalarAbi[T]
) extends KernelParam,
      Expr[T]:
  override type Value = T
  require(
    valueType.sizeBytes == scalarAbi.abiType.sizeBytes &&
      valueType.alignmentBytes == scalarAbi.abiType.alignmentBytes,
    s"$name has inconsistent CUDA type and ABI metadata"
  )

final case class BufferParam[T, Mode <: AccessMode](
    name: String,
    valueType: CudaType[T]
)(using accessMode: AccessModeWitness[Mode]
) extends KernelParam:
  override type Value = T
  def access: BufferAccess = accessMode.access
