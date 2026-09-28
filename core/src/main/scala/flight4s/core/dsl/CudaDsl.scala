package flight4s.core.dsl

import scala.collection.mutable.ArrayBuffer
import scala.annotation.targetName

import flight4s.core.abi.ScalarAbi
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock}
import flight4s.core.types.*

object CudaDsl:
  final class BlockBuilder private[dsl] (
      private val sharedDeclarations: Option[ArrayBuffer[SharedArray[?, ?]]],
      private val collectiveBlocks: ArrayBuffer[BlockShapeRequirement] = ArrayBuffer.empty
  ):
    private val statements = ArrayBuffer.empty[Stmt]

    private[dsl] def append(statement: Stmt): Unit =
      ExpressionStaging.requireStatementsAllowed(statement.span)
      statements += statement

    private[dsl] def declareShared(memory: SharedArray[?, ?]): Unit =
      ExpressionStaging.requireStatementsAllowed(memory.span)
      sharedDeclarations match
        case Some(declarations) =>
          declarations += memory
        case None =>
          throw DslError(
            DslErrorCode.SharedMemoryDeclarationOutsideKernelBody,
            "shared memory must be declared directly in a kernel body",
            memory.span
          )

    private[dsl] def nested(): BlockBuilder =
      BlockBuilder(None, collectiveBlocks)

    private[dsl] def requireBlock(shape: LaunchBlock, span: SourceSpan): Unit =
      ExpressionStaging.requireStatementsAllowed(span)
      collectiveBlocks += BlockShapeRequirement(shape, span)

    private[dsl] def blockRequirements: Vector[BlockShapeRequirement] =
      collectiveBlocks.toVector.distinct

    private[dsl] def result(): Block =
      Block(statements.toVector)

    private[dsl] def sharedMemory: Vector[SharedArray[?, ?]] =
      sharedDeclarations.fold(Vector.empty)(_.toVector)

  def literal[T](value: T)(using valueType: CudaType[T]): Expr[T] =
    Literal(value, valueType)

  def float2(x: Expr[Float], y: Expr[Float])(using position: DslSourcePosition): Expr[Float2] =
    FloatVectorConstruct(Vector(x, y), F32x2, position.span)

  def float4(x: Expr[Float], y: Expr[Float], z: Expr[Float], w: Expr[Float])(using
      position: DslSourcePosition
  ): Expr[Float4] = FloatVectorConstruct(Vector(x, y, z, w), F32x4, position.span)

  extension [T](vector: Expr[T])
    def x(using vectorType: FloatVectorType[T], position: DslSourcePosition): Expr[Float] =
      FloatVectorComponent(vector, 0, vectorType, position.span)

    def y(using vectorType: FloatVectorType[T], position: DslSourcePosition): Expr[Float] =
      FloatVectorComponent(vector, 1, vectorType, position.span)

    def map(f: Expr[Float] => Expr[Float])(using
        vectorType: FloatVectorType[T], position: DslSourcePosition
    ): Expr[T] =
      FloatVectorConstruct(Vector.tabulate(vectorType.componentCount) { index =>
        ExpressionStaging.expression(f(FloatVectorComponent(vector, index, vectorType, position.span)))
      }, vectorType, position.span)

    def zipWith(other: Expr[T])(f: (Expr[Float], Expr[Float]) => Expr[Float])(using
        vectorType: FloatVectorType[T], position: DslSourcePosition
    ): Expr[T] =
      FloatVectorConstruct(Vector.tabulate(vectorType.componentCount) { index =>
        ExpressionStaging.expression(f(FloatVectorComponent(vector, index, vectorType, position.span),
          FloatVectorComponent(other, index, vectorType, position.span)))
      }, vectorType, position.span)

  extension (vector: Expr[Float4])
    def z(using position: DslSourcePosition): Expr[Float] = FloatVectorComponent(vector, 2, F32x4, position.span)
    def w(using position: DslSourcePosition): Expr[Float] = FloatVectorComponent(vector, 3, F32x4, position.span)

  def choose[T](condition: Expr[Boolean])(whenTrue: => Expr[T])(whenFalse: => Expr[T])(using
      valueType: CudaType[T],
      position: DslSourcePosition
  ): Expr[T] =
    Conditional(
      condition,
      ExpressionStaging.expression(whenTrue),
      ExpressionStaging.expression(whenFalse),
      valueType,
      position.span
    )

  def exp[T](value: Expr[T])(using mathType: FloatingMathType[T], position: DslSourcePosition): Expr[T] =
    UnaryMath(UnaryMathOperator.Exp, value, mathType, position.span)

  def log[T](value: Expr[T])(using mathType: FloatingMathType[T], position: DslSourcePosition): Expr[T] =
    UnaryMath(UnaryMathOperator.Log, value, mathType, position.span)

  def sqrt[T](value: Expr[T])(using mathType: FloatingMathType[T], position: DslSourcePosition): Expr[T] =
    UnaryMath(UnaryMathOperator.Sqrt, value, mathType, position.span)

  def rsqrt[T](value: Expr[T])(using mathType: FloatingMathType[T], position: DslSourcePosition): Expr[T] =
    UnaryMath(UnaryMathOperator.Rsqrt, value, mathType, position.span)

  def tanh[T](value: Expr[T])(using mathType: FloatingMathType[T], position: DslSourcePosition): Expr[T] =
    UnaryMath(UnaryMathOperator.Tanh, value, mathType, position.span)

  def input[T](name: String)(using valueType: CudaType[T]): BufferParam[T, ReadOnly] =
    BufferParam(name, valueType)

  def in[T](name: String)(using valueType: CudaType[T]): BufferParam[T, ReadOnly] =
    input(name)

  def output[T](name: String)(using valueType: CudaType[T]): BufferParam[T, ReadWrite] =
    BufferParam(name, valueType)

  def out[T](name: String)(using valueType: CudaType[T]): BufferParam[T, ReadWrite] =
    output(name)

  def inOut[T](name: String)(using valueType: CudaType[T]): BufferParam[T, ReadWrite] =
    BufferParam(name, valueType)

  def value[T](name: String)(using
      valueType: CudaType[T],
      scalarAbi: ScalarAbi[T]
  ): ScalarParam[T] =
    ScalarParam(name, valueType)

  def params()
      : KernelSignature[EmptyTuple] { type Bindings = EmptyTuple } =
    KernelSignature.fromTuple(EmptyTuple)

  def params[Params <: Tuple](
      bindings: Params
  )(using tuple: KernelParamTuple[Params])
      : KernelSignature[KernelArgumentsOf[Params]] { type Bindings = Params } =
    KernelSignature.fromTuple(bindings)

  def params[P1 <: KernelParam](
      p1: P1
  )(using KernelParamAbi[P1])
      : KernelSignature[KernelArgumentOf[P1] *: EmptyTuple] {
    type Bindings = P1 *: EmptyTuple
  } =
    KernelSignature.fromTuple(p1 *: EmptyTuple)

  def params[P1 <: KernelParam, P2 <: KernelParam](
      p1: P1,
      p2: P2
  )(using KernelParamAbi[P1], KernelParamAbi[P2])
      : KernelSignature[
        KernelArgumentOf[P1] *: KernelArgumentOf[P2] *: EmptyTuple
      ] {
    type Bindings = P1 *: P2 *: EmptyTuple
  } =
    KernelSignature.fromTuple(p1 *: p2 *: EmptyTuple)

  def params[P1 <: KernelParam, P2 <: KernelParam, P3 <: KernelParam](
      p1: P1,
      p2: P2,
      p3: P3
  )(using KernelParamAbi[P1], KernelParamAbi[P2], KernelParamAbi[P3])
      : KernelSignature[
    KernelArgumentOf[P1] *: KernelArgumentOf[P2] *:
      KernelArgumentOf[P3] *: EmptyTuple
  ] {
    type Bindings = P1 *: P2 *: P3 *: EmptyTuple
  } =
    KernelSignature.fromTuple(p1 *: p2 *: p3 *: EmptyTuple)

  def params[
      P1 <: KernelParam,
      P2 <: KernelParam,
      P3 <: KernelParam,
      P4 <: KernelParam
  ](
      p1: P1,
      p2: P2,
      p3: P3,
      p4: P4
  )(using
      KernelParamAbi[P1],
      KernelParamAbi[P2],
      KernelParamAbi[P3],
      KernelParamAbi[P4]
  ): KernelSignature[
    KernelArgumentOf[P1] *: KernelArgumentOf[P2] *:
      KernelArgumentOf[P3] *: KernelArgumentOf[P4] *: EmptyTuple
  ] {
    type Bindings = P1 *: P2 *: P3 *: P4 *: EmptyTuple
  } =
    KernelSignature.fromTuple(p1 *: p2 *: p3 *: p4 *: EmptyTuple)

  def params[
      P1 <: KernelParam,
      P2 <: KernelParam,
      P3 <: KernelParam,
      P4 <: KernelParam,
      P5 <: KernelParam
  ](
      p1: P1,
      p2: P2,
      p3: P3,
      p4: P4,
      p5: P5
  )(using
      KernelParamAbi[P1],
      KernelParamAbi[P2],
      KernelParamAbi[P3],
      KernelParamAbi[P4],
      KernelParamAbi[P5]
  ): KernelSignature[
    KernelArgumentOf[P1] *: KernelArgumentOf[P2] *:
      KernelArgumentOf[P3] *: KernelArgumentOf[P4] *:
      KernelArgumentOf[P5] *: EmptyTuple
  ] {
    type Bindings = P1 *: P2 *: P3 *: P4 *: P5 *: EmptyTuple
  } =
    KernelSignature.fromTuple(p1 *: p2 *: p3 *: p4 *: p5 *: EmptyTuple)

  def params[
      P1 <: KernelParam,
      P2 <: KernelParam,
      P3 <: KernelParam,
      P4 <: KernelParam,
      P5 <: KernelParam,
      P6 <: KernelParam
  ](
      p1: P1,
      p2: P2,
      p3: P3,
      p4: P4,
      p5: P5,
      p6: P6
  )(using
      KernelParamAbi[P1],
      KernelParamAbi[P2],
      KernelParamAbi[P3],
      KernelParamAbi[P4],
      KernelParamAbi[P5],
      KernelParamAbi[P6]
  ): KernelSignature[
    KernelArgumentOf[P1] *: KernelArgumentOf[P2] *:
      KernelArgumentOf[P3] *: KernelArgumentOf[P4] *:
      KernelArgumentOf[P5] *: KernelArgumentOf[P6] *: EmptyTuple
  ] {
    type Bindings = P1 *: P2 *: P3 *: P4 *: P5 *: P6 *: EmptyTuple
  } =
    KernelSignature.fromTuple(p1 *: p2 *: p3 *: p4 *: p5 *: p6 *: EmptyTuple)

  def paramsTuple[Params <: Tuple](
      bindings: Params
  )(using tuple: KernelParamTuple[Params])
      : KernelSignature[KernelArgumentsOf[Params]] { type Bindings = Params } =
    params(bindings)

  def kernel[Args <: Tuple](
      name: String,
      signature: KernelSignature[Args]
  )(
      body: signature.Bindings => (BlockBuilder ?=> Unit)
  ): Kernel[Args] =
    val builder = BlockBuilder(Some(ArrayBuffer.empty))
    body(signature.bindings)(using builder)
    Kernel(
      KernelIR(
        name,
        signature,
        builder.result(),
        builder.sharedMemory,
        requiredBlock = builder.blockRequirements.headOption.map(_.shape),
        blockRequirements = builder.blockRequirements
      )
    )

  def kernel(
      name: String
  )(body: BlockBuilder ?=> Unit): Kernel[EmptyTuple] =
    val signature = params()
    kernel(name, signature)(_ => body)

  def reduceSum[Input, Accumulator](
      indexName: String,
      from: Expr[Int],
      until: Expr[Int],
      initial: Expr[Accumulator],
      policy: ReductionPolicy = ReductionPolicy.Strict,
      step: Int = 1
  )(
      body: Expr[Int] => Expr[Input]
  )(using
      rule: AccumulatorType[Input, Accumulator],
      addition: AdditiveType[Accumulator],
      position: DslSourcePosition
  ): Expr[Accumulator] =
    val index = ReductionIndex(indexName, position.span)
    ReduceSum(
      index = index,
      from = from,
      until = until,
      initial = initial,
      value = ExpressionStaging.expression(body(index)),
      rule = rule,
      addition = addition,
      policy = policy,
      span = position.span,
      step = step
    )

  def gpuRange(
      indexName: String,
      from: Expr[Int],
      until: Expr[Int]
  ): GpuRange =
    new GpuRange(indexName, from, until)

  def local[T](
      name: String,
      initial: Expr[T]
  )(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): LocalVariable[T] =
    val variable = LocalVariable(name, valueType, position.span)
    builder.append(LocalDeclaration(variable, initial, position.span))
    variable

  /** Evaluates once at this statement position and exposes only the stored value. */
  def let[T](name: String, initial: Expr[T])(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): Expr[T] =
    Load(local(name, initial), position.span)

  def localArray[T](
      name: String,
      elementCount: Int
  )(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): LocalArray[T] =
    val array = LocalArray(name, valueType, elementCount, position.span)
    builder.append(LocalArrayDeclaration(array, position.span))
    array

  def sharedArray[T](
      name: String,
      elementCount: Int
  )(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): SharedArray[T, Rank1] =
    val memory: SharedArray[T, Rank1] =
      SharedArray(
        name,
        valueType,
        StaticSharedMemory(elementCount),
        position.span
      )
    builder.declareShared(memory)
    memory

  def sharedArray2D[T](
      name: String,
      rows: Int,
      columns: Int
  )(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): SharedArray[T, Rank2] =
    sharedArray2D(name, rows, columns, columns)

  def sharedArray2D[T](
      name: String,
      rows: Int,
      columns: Int,
      rowStride: Int
  )(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): SharedArray[T, Rank2] =
    val memory: SharedArray[T, Rank2] =
      SharedArray(
        name,
        valueType,
        StaticSharedMemory.twoDimensional(rows, columns, rowStride),
        position.span
      )
    builder.declareShared(memory)
    memory

  def dynamicSharedArray[T](
      name: String
  )(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): SharedArray[T, Rank1] =
    val memory: SharedArray[T, Rank1] =
      SharedArray(name, valueType, DynamicSharedMemory, position.span)
    builder.declareShared(memory)
    memory

  def sharedArray3D[T](
      name: String,
      depth: Int,
      rows: Int,
      columns: Int
  )(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): SharedArray[T, Rank3] =
    sharedArray3D(name, depth, rows, columns, columns)

  def sharedArray3D[T](
      name: String,
      depth: Int,
      rows: Int,
      columns: Int,
      rowStride: Int
  )(using
      valueType: CudaType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): SharedArray[T, Rank3] =
    val memory: SharedArray[T, Rank3] =
      SharedArray(
        name,
        valueType,
        StaticSharedMemory.threeDimensional(
          depth,
          rows,
          columns,
          rowStride
        ),
        position.span
      )
    builder.declareShared(memory)
    memory

  def constantArray[T](
      name: String,
      elementCount: Int
  )(using
      valueType: CudaType[T],
      position: DslSourcePosition
  ): ConstantArray[T] =
    ConstantArray(name, valueType, elementCount, position.span)

  def module(
      constants: Iterable[ConstantArray[?]] = Vector.empty,
      kernels: Iterable[Kernel[?]] = Vector.empty
  ): CudaModuleIR =
    CudaModuleIR(
      constants.toVector,
      kernels.iterator.map(_.ir).toVector
    )

  /** Executes the update once and binds CUDA's returned old value to a device local. */
  def atomicFetchAdd[T, Space <: Global | Shared](
      name: String,
      target: Place[T, Space, ReadWrite],
      value: Expr[T]
  )(using
      addition: AtomicAddType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): Expr[T] =
    val result = LocalVariable(name, addition, position.span)
    builder.append(AtomicFetchAdd(result, target, value, addition, position.span))
    Load(result, position.span)

  def atomicAdd[T, Space <: Global | Shared](
      target: Place[T, Space, ReadWrite],
      value: Expr[T]
  )(using
      addition: AtomicAddType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): Unit =
    builder.append(AtomicAdd(target, value, addition, position.span))

  def accumulate[T](
      target: LocalVariable[T],
      value: Expr[T]
  )(using
      addition: AdditiveType[T],
      builder: BlockBuilder,
      position: DslSourcePosition
  ): Unit =
    builder.append(Accumulate(target, value, addition, position.span))

  def gpuFor(
      indexName: String,
      from: Expr[Int],
      until: Expr[Int],
      step: Int = 1
  )(
      body: Expr[Int] => (BlockBuilder ?=> Unit)
  )(using parent: BlockBuilder, position: DslSourcePosition): Unit =
    val index = LoopIndex(indexName, position.span)
    val nested = parent.nested()
    body(index)(using nested)
    parent.append(
      ForLoop(index, from, until, nested.result(), position.span, step)
    )

  def scoped(body: BlockBuilder ?=> Unit)(using
      parent: BlockBuilder,
      position: DslSourcePosition
  ): Unit =
    val nested = parent.nested()
    body(using nested)
    parent.append(ScopedBlock(nested.result(), position.span))

  def when(condition: Expr[Boolean])(
      body: BlockBuilder ?=> Unit
  )(using parent: BlockBuilder, position: DslSourcePosition): Unit =
    val nested = parent.nested()
    body(using nested)
    parent.append(IfThen(condition, nested.result(), span = position.span))

  def gpuIf(condition: Expr[Boolean])(
      thenBody: BlockBuilder ?=> Unit
  )(
      elseBody: BlockBuilder ?=> Unit
  )(using parent: BlockBuilder, position: DslSourcePosition): Unit =
    val thenBuilder = parent.nested()
    thenBody(using thenBuilder)
    val elseBuilder = parent.nested()
    elseBody(using elseBuilder)
    parent.append(
      IfThen(
        condition = condition,
        thenBlock = thenBuilder.result(),
        elseBlock = Some(elseBuilder.result()),
        span = position.span
      )
    )

  def barrier()(using
      builder: BlockBuilder,
      position: DslSourcePosition
  ): Unit =
    builder.append(Barrier(position.span))

  object bits:
    def popCount[T](value: Expr[T])(using wordType: BitwiseType[T], position: DslSourcePosition): Expr[Int] =
      PopulationCount(value, wordType, position.span)

  object block:
    def reduction[T](name: String, shape: LaunchBlock)(using
        valueType: CudaType[T], builder: BlockBuilder, position: DslSourcePosition
    ): BlockReduction[T] =
      val count = BigInt(shape.x) * shape.y * shape.z
      if count > 1024 || (count & (count - 1)) != 0 then
        throw DslError(DslErrorCode.InvalidBlockReductionShape,
          "block reduction requires a power-of-two thread count from 1 through 1024", position.span)
      new BlockReduction(sharedArray[T](name, count.toInt), shape, count.toInt, valueType)

  object warp:
    def reduceSum[T](name: String, mask: UInt, value: Expr[T], width: Int = 32)(using
        WarpShuffleType[T], AdditiveType[T], BlockBuilder, DslSourcePosition
    ): Expr[T] = reduceTree(name, mask, value, width)(_ + _)

    def reduceTree[T](name: String, mask: UInt, value: Expr[T], width: Int = 32)(
        combine: (Expr[T], Expr[T]) => Expr[T]
    )(using shuffleType: WarpShuffleType[T], builder: BlockBuilder, position: DslSourcePosition): Expr[T] =
      validateReductionGroup(mask, width)
      if width == 1 then let(name, value)
      else
        val linearThread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
        var current = let(s"${name}_input", value)
        var distance = 1
        while distance < width do
          val partner = shuffleXor(s"${name}_partner_$distance", literal(mask), current, literal(distance), width)
          val lowerLane = (linearThread & literal(distance)) === literal(0)
          // Preserve left/right subtree order in every lane, including noncommutative combines.
          val left = choose(lowerLane)(current)(partner)
          val right = choose(lowerLane)(partner)(current)
          val combined = ExpressionStaging.expression(combine(left, right))
          val stageName = if distance * 2 == width then name else s"${name}_stage_$distance"
          current = let(stageName, combined)
          distance *= 2
        current

    private def validateReductionGroup(mask: UInt, width: Int)(using position: DslSourcePosition): Unit =
      val validWidth = width >= 1 && width <= 32 && (width & (width - 1)) == 0
      val bits = java.lang.Integer.toUnsignedLong(mask.toIntBits)
      val completeGroups = validWidth && (0 until 32 by width).forall { start =>
        val group = ((1L << width) - 1L) << start
        val selected = bits & group
        selected == 0L || selected == group
      }
      if bits == 0L || !completeGroups then
        throw DslError(DslErrorCode.InvalidWarpReductionGroup,
          "warp reduction requires a nonempty static mask containing complete aligned groups " +
            "of width 1, 2, 4, 8, 16, or 32", position.span)

    def sync(mask: Expr[UInt])(using builder: BlockBuilder, position: DslSourcePosition): Unit =
      builder.append(WarpBarrier(mask, position.span))

    def shuffle[T](
        name: String,
        mask: Expr[UInt],
        value: Expr[T],
        sourceLane: Expr[Int],
        width: Int = 32
    )(using shuffleType: WarpShuffleType[T], builder: BlockBuilder, position: DslSourcePosition): Expr[T] =
      shuffleValue(name, mask, value, WarpShuffleOperator.Direct, sourceLane, width)

    def shuffleUp[T](
        name: String, mask: Expr[UInt], value: Expr[T], delta: Expr[UInt], width: Int = 32
    )(using WarpShuffleType[T], BlockBuilder, DslSourcePosition): Expr[T] =
      shuffleValue(name, mask, value, WarpShuffleOperator.Up, delta, width)

    def shuffleDown[T](
        name: String, mask: Expr[UInt], value: Expr[T], delta: Expr[UInt], width: Int = 32
    )(using WarpShuffleType[T], BlockBuilder, DslSourcePosition): Expr[T] =
      shuffleValue(name, mask, value, WarpShuffleOperator.Down, delta, width)

    def shuffleXor[T](
        name: String, mask: Expr[UInt], value: Expr[T], laneMask: Expr[Int], width: Int = 32
    )(using WarpShuffleType[T], BlockBuilder, DslSourcePosition): Expr[T] =
      shuffleValue(name, mask, value, WarpShuffleOperator.Xor, laneMask, width)

    private def shuffleValue[T, S](
        name: String, mask: Expr[UInt], value: Expr[T], operator: WarpShuffleOperator[S],
        selector: Expr[S], width: Int
    )(using shuffleType: WarpShuffleType[T], builder: BlockBuilder, position: DslSourcePosition): Expr[T] =
      val result = LocalVariable(name, shuffleType, position.span)
      builder.append(WarpShuffle(result, mask, value, operator, selector, width, shuffleType, position.span))
      Load(result, position.span)

    def ballot(name: String, mask: Expr[UInt], predicate: Expr[Boolean])(using
        BlockBuilder, DslSourcePosition
    ): Expr[UInt] = vote(name, WarpVoteOperator.Ballot, mask, predicate)

    def all(name: String, mask: Expr[UInt], predicate: Expr[Boolean])(using
        BlockBuilder, DslSourcePosition
    ): Expr[Boolean] = vote(name, WarpVoteOperator.All, mask, predicate)

    def any(name: String, mask: Expr[UInt], predicate: Expr[Boolean])(using
        BlockBuilder, DslSourcePosition
    ): Expr[Boolean] = vote(name, WarpVoteOperator.Any, mask, predicate)

    private def vote[T](name: String, operator: WarpVoteOperator[T], mask: Expr[UInt], predicate: Expr[Boolean])(using
        builder: BlockBuilder, position: DslSourcePosition
    ): Expr[T] =
      val result = LocalVariable(name, operator.resultType, position.span)
      builder.append(WarpVote(result, operator, mask, predicate, position.span))
      Load(result, position.span)

  object threadIdx:
    def x: Expr[Int] = Intrinsic("threadIdx.x", I32)
    def y: Expr[Int] = Intrinsic("threadIdx.y", I32)
    def z: Expr[Int] = Intrinsic("threadIdx.z", I32)

  object blockIdx:
    def x: Expr[Int] = Intrinsic("blockIdx.x", I32)
    def y: Expr[Int] = Intrinsic("blockIdx.y", I32)
    def z: Expr[Int] = Intrinsic("blockIdx.z", I32)

  object blockDim:
    def x: Expr[Int] = Intrinsic("blockDim.x", I32)
    def y: Expr[Int] = Intrinsic("blockDim.y", I32)
    def z: Expr[Int] = Intrinsic("blockDim.z", I32)

  object gridDim:
    def x: Expr[Int] = Intrinsic("gridDim.x", I32)
    def y: Expr[Int] = Intrinsic("gridDim.y", I32)
    def z: Expr[Int] = Intrinsic("gridDim.z", I32)

  extension (value: Expr[Int])
    @targetName("signedShiftLeft")
    def <<(distance: Expr[Int])(using position: DslSourcePosition): Expr[Int] =
      SignedShift(SignedShiftOperator.Left, value, distance, position.span)

    @targetName("signedShiftRight")
    def >>(distance: Expr[Int])(using position: DslSourcePosition): Expr[Int] =
      SignedShift(SignedShiftOperator.ArithmeticRight, value, distance, position.span)

    @targetName("signedLogicalShiftRight")
    def >>>(distance: Expr[Int])(using position: DslSourcePosition): Expr[Int] =
      SignedShift(SignedShiftOperator.LogicalRight, value, distance, position.span)

  extension (value: Expr[UInt])
    def <<(distance: Expr[Int])(using position: DslSourcePosition): Expr[UInt] =
      UnsignedShift(UnsignedShiftOperator.Left, value, distance, position.span)

    def >>(distance: Expr[Int])(using position: DslSourcePosition): Expr[UInt] =
      UnsignedShift(UnsignedShiftOperator.Right, value, distance, position.span)

    def >>>(distance: Expr[Int])(using position: DslSourcePosition): Expr[UInt] =
      UnsignedShift(UnsignedShiftOperator.Right, value, distance, position.span)

  extension [T](left: Expr[T])
    def unary_~(using valueType: BitwiseType[T], position: DslSourcePosition): Expr[T] =
      Binary(BinaryOperator.BitXor, left, Literal(valueType.allBitsSet, valueType, position.span), valueType, position.span)

    def +(right: Expr[T])(using valueType: AdditiveType[T]): Expr[T] =
      Binary(BinaryOperator.Add, left, right, valueType)

    def -(right: Expr[T])(using valueType: AdditiveType[T]): Expr[T] =
      Binary(BinaryOperator.Subtract, left, right, valueType)

    def *(right: Expr[T])(using valueType: MultiplicativeType[T]): Expr[T] =
      Binary(BinaryOperator.Multiply, left, right, valueType)

    def /(right: Expr[T])(using valueType: DivisibleType[T]): Expr[T] =
      Binary(BinaryOperator.Divide, left, right, valueType)

    def %(right: Expr[T])(using valueType: RemainderType[T]): Expr[T] =
      Binary(BinaryOperator.Remainder, left, right, valueType)

    def &(right: Expr[T])(using valueType: BitwiseType[T], position: DslSourcePosition): Expr[T] =
      Binary(BinaryOperator.BitAnd, left, right, valueType, position.span)

    def |(right: Expr[T])(using valueType: BitwiseType[T], position: DslSourcePosition): Expr[T] =
      Binary(BinaryOperator.BitOr, left, right, valueType, position.span)

    def ^(right: Expr[T])(using valueType: BitwiseType[T], position: DslSourcePosition): Expr[T] =
      Binary(BinaryOperator.BitXor, left, right, valueType, position.span)

    def <(right: Expr[T])(using valueType: OrderedType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.LessThan, left, right, valueType)

    def <=(right: Expr[T])(using valueType: OrderedType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.LessThanOrEqual, left, right, valueType)

    def >(right: Expr[T])(using valueType: OrderedType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.GreaterThan, left, right, valueType)

    def >=(right: Expr[T])(using valueType: OrderedType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.GreaterThanOrEqual, left, right, valueType)

    def ===(right: Expr[T])(using valueType: EqualityComparableType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.Equal, left, right, valueType)

    def !==(right: Expr[T])(using valueType: EqualityComparableType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.NotEqual, left, right, valueType)

    def toAccumulator[A](using rule: AccumulatorType[T, A]): Expr[A] =
      ToAccumulator(left, rule)

  extension (left: Expr[Boolean])
    def &&(right: => Expr[Boolean])(using DslSourcePosition): Expr[Boolean] =
      choose(left)(right)(literal(false))

    def ||(right: => Expr[Boolean])(using DslSourcePosition): Expr[Boolean] =
      choose(left)(literal(true))(right)

    def unary_!(using DslSourcePosition): Expr[Boolean] =
      choose(left)(literal(false))(literal(true))

  object convert:
    def f64ToF32(value: Expr[Double], rounding: RoundingMode = RoundingMode.NearestEven)(using
        position: DslSourcePosition
    ): Expr[Float] =
      Convert(value, F32, rounding, SaturationMode.NoSaturation, position.span)

    def f32ToF64(value: Expr[Float])(using position: DslSourcePosition): Expr[Double] =
      Convert(value, F64, RoundingMode.NearestEven, SaturationMode.NoSaturation, position.span)

    def i32ToF32(value: Expr[Int], rounding: RoundingMode = RoundingMode.NearestEven)(using
        position: DslSourcePosition
    ): Expr[Float] =
      Convert(value, F32, rounding, SaturationMode.NoSaturation, position.span)

    def u32ToF32(value: Expr[UInt], rounding: RoundingMode = RoundingMode.NearestEven)(using
        position: DslSourcePosition
    ): Expr[Float] =
      Convert(value, F32, rounding, SaturationMode.NoSaturation, position.span)

    def i32ToF64(value: Expr[Int])(using position: DslSourcePosition): Expr[Double] =
      Convert(value, F64, RoundingMode.NearestEven, SaturationMode.NoSaturation, position.span)

    def u32ToF64(value: Expr[UInt])(using position: DslSourcePosition): Expr[Double] =
      Convert(value, F64, RoundingMode.NearestEven, SaturationMode.NoSaturation, position.span)

    def f32ToF16(
        value: Expr[Float],
        rounding: RoundingMode = RoundingMode.NearestEven
    ): Expr[Float16] =
      Convert(
        value = value,
        valueType = F16,
        rounding = rounding,
        saturation = SaturationMode.NoSaturation
      )

    def f16ToF32(value: Expr[Float16]): Expr[Float] =
      Convert(
        value = value,
        valueType = F32,
        rounding = RoundingMode.NearestEven,
        saturation = SaturationMode.NoSaturation
      )

    def f32ToBF16(
        value: Expr[Float],
        rounding: RoundingMode = RoundingMode.NearestEven
    ): Expr[BFloat16] =
      Convert(
        value = value,
        valueType = BF16,
        rounding = rounding,
        saturation = SaturationMode.NoSaturation
      )

    def bf16ToF32(value: Expr[BFloat16]): Expr[Float] =
      Convert(
        value = value,
        valueType = F32,
        rounding = RoundingMode.NearestEven,
        saturation = SaturationMode.NoSaturation
      )

    def f32ToFP8E4M3(
        value: Expr[Float],
        saturation: SaturationMode = SaturationMode.SaturateFinite
    ): Expr[Float8E4M3] =
      Convert(
        value = value,
        valueType = FP8E4M3,
        rounding = RoundingMode.NearestEven,
        saturation = saturation
      )

    def fp8E4M3ToF32(value: Expr[Float8E4M3]): Expr[Float] =
      Convert(
        value = value,
        valueType = F32,
        rounding = RoundingMode.NearestEven,
        saturation = SaturationMode.NoSaturation
      )

    def f32ToFP8E5M2(
        value: Expr[Float],
        saturation: SaturationMode = SaturationMode.SaturateFinite
    ): Expr[Float8E5M2] =
      Convert(
        value = value,
        valueType = FP8E5M2,
        rounding = RoundingMode.NearestEven,
        saturation = saturation
      )

    def fp8E5M2ToF32(value: Expr[Float8E5M2]): Expr[Float] =
      Convert(
        value = value,
        valueType = F32,
        rounding = RoundingMode.NearestEven,
        saturation = SaturationMode.NoSaturation
      )

  extension [T, Mode <: AccessMode](buffer: BufferParam[T, Mode])
    def apply(index: Expr[Int]): BufferElement[T, Mode] =
      BufferElement(buffer.name, index, buffer.valueType)

  extension [T](array: ConstantArray[T])
    def apply(index: Expr[Int]): ConstantElement[T] =
      ConstantElement(array.name, index, array.valueType)

  extension [T](array: SharedArray[T, Rank1])
    def apply(index: Expr[Int]): SharedElement[T] =
      SharedElement(array.name, Vector(index), array.valueType)

  extension [T](array: SharedArray[T, Rank2])
    def apply(row: Expr[Int], column: Expr[Int]): SharedElement[T] =
      SharedElement(array.name, Vector(row, column), array.valueType)

  extension [T](array: SharedArray[T, Rank3])
    def apply(
        depth: Expr[Int],
        row: Expr[Int],
        column: Expr[Int]
    ): SharedElement[T] =
      SharedElement(
        array.name,
        Vector(depth, row, column),
        array.valueType
      )

  extension [T](array: LocalArray[T])
    def apply(index: Expr[Int]): LocalArrayElement[T] =
      LocalArrayElement(array.name, index, array.valueType)

  extension [T, Space <: AddressSpace, Mode <: AccessMode](
      place: Place[T, Space, Mode]
  )
    def read: Expr[T] =
      Load(place)

  extension [T, Space <: AddressSpace](
      place: Place[T, Space, ReadWrite]
  )
    infix def :=(value: Expr[T])(using
        builder: BlockBuilder,
        position: DslSourcePosition
    ): Unit =
      builder.append(Store(place, value, position.span))
