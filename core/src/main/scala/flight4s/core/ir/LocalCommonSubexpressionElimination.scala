package flight4s.core.ir

import scala.collection.mutable

import flight4s.core.types.I32

private[core] object LocalCommonSubexpressionElimination:
  private enum IntegerExpressionKey:
    case Literal(value: Int)
    case Scalar(name: String)
    case Intrinsic(name: String)
    case LoopIndex(name: String)
    case ReductionIndex(name: String)
    case Binary(
        operator: BinaryOperator,
        left: IntegerExpressionKey,
        right: IntegerExpressionKey
    )

  private final class FreshNames(initiallyOccupied: Set[String]):
    private val occupied = mutable.Set.from(initiallyOccupied)
    private var nextId = 0

    def cse(): String =
      var candidate = s"flight4s_cse_$nextId"
      nextId += 1
      while occupied.contains(candidate) do
        candidate = s"flight4s_cse_$nextId"
        nextId += 1
      occupied += candidate
      candidate

  def kernel[Args <: Tuple](
      kernel: KernelIR[Args],
      reservedNames: Set[String] = Set.empty
  ): KernelIR[Args] =
    val freshNames = FreshNames(reservedNames ++ collectNames(kernel))
    kernel.copy(body = transformBlock(kernel.body, freshNames))

  def block(block: Block): Block =
    transformBlock(block, FreshNames(collectNames(block)))

  private def transformBlock(block: Block, freshNames: FreshNames): Block =
    Block(block.statements.flatMap(transformStatement(_, freshNames)))

  private def transformStatement(
      statement: Stmt,
      freshNames: FreshNames
  ): Vector[Stmt] =
    val (declarations, rewritten) = eliminateOwnExpressions(statement, freshNames)
    val nested = rewritten match
      case branch: IfThen =>
        branch.copy(
          thenBlock = transformBlock(branch.thenBlock, freshNames),
          elseBlock = branch.elseBlock.map(transformBlock(_, freshNames))
        )
      case scoped: ScopedBlock =>
        scoped.copy(body = transformBlock(scoped.body, freshNames))
      case loop: ForLoop =>
        loop.copy(body = transformBlock(loop.body, freshNames))
      case other => other
    declarations :+ nested

  private def eliminateOwnExpressions(
      statement: Stmt,
      freshNames: FreshNames
  ): (Vector[Stmt], Stmt) =
    val counts = mutable.Map.empty[IntegerExpressionKey, Int]
    ownExpressions(statement).foreach(collectCounts(_, counts))
    val repeated = counts.collect { case (key, count) if count > 1 => key }.toSet

    if repeated.isEmpty then (Vector.empty, statement)
    else
      val rewriter = ExpressionRewriter(repeated, freshNames, statement.span)
      val rewritten = statement match
        case declaration: LocalDeclaration[?] =>
          declaration.copy(initial = rewriter.expression(declaration.initial))
        case vote: WarpVote[?] =>
          vote.copy(mask = rewriter.expression(vote.mask), predicate = rewriter.expression(vote.predicate))
        case store: Store[?, ?] =>
          rewriteStore(store, rewriter)
        case atomic: AtomicAdd[?, ?] =>
          rewriteAtomic(atomic, rewriter)
        case atomic: AtomicFetchAdd[?, ?] =>
          rewriteAtomicFetch(atomic, rewriter)
        case accumulation: Accumulate[?] =>
          accumulation.copy(value = rewriter.expression(accumulation.value))
        case branch: IfThen =>
          branch.copy(condition = rewriter.expression(branch.condition))
        case loop: ForLoop =>
          loop.copy(
            from = rewriter.expression(loop.from),
            until = rewriter.expression(loop.until)
          )
        case _: LocalArrayDeclaration[?] | _: ScopedBlock | _: Barrier =>
          statement
      (rewriter.declarations, rewritten)

  private final class ExpressionRewriter(
      repeated: Set[IntegerExpressionKey],
      freshNames: FreshNames,
      fallbackSpan: SourceSpan
  ):
    private val bindings = mutable.LinkedHashMap.empty[IntegerExpressionKey, LocalVariable[Int]]
    private val generated = Vector.newBuilder[Stmt]

    def declarations: Vector[Stmt] = generated.result()

    def expression[T](expression: Expr[T]): Expr[T] =
      candidateKey(expression) match
        case Some(key) if repeated.contains(key) =>
          val local = bindings.getOrElseUpdate(
            key,
            {
              val declarationSpan =
                if expression.span != SourceSpan.Unknown then expression.span
                else fallbackSpan
              val variable = LocalVariable(freshNames.cse(), I32, declarationSpan)
              generated += LocalDeclaration(
                variable,
                expression.asInstanceOf[Expr[Int]],
                declarationSpan
              )
              variable
            }
          )
          Load(local, expression.span).asInstanceOf[Expr[T]]
        case _ => rewriteChildren(expression)

    def place[
        T,
        Space <: AddressSpace,
        Mode <: AccessMode
    ](place: Place[T, Space, Mode]): Place[T, Space, Mode] = place match
      case buffer: BufferElement[?, ?] =>
        buffer
          .copy(index = expression(buffer.index))
          .asInstanceOf[Place[T, Space, Mode]]
      case constant: ConstantElement[?] =>
        constant
          .copy(index = expression(constant.index))
          .asInstanceOf[Place[T, Space, Mode]]
      case shared: SharedElement[?] =>
        shared
          .copy(indices = shared.indices.map(expression))
          .asInstanceOf[Place[T, Space, Mode]]
      case local: LocalVariable[?] =>
        local.asInstanceOf[Place[T, Space, Mode]]
      case localArray: LocalArrayElement[?] =>
        localArray
          .copy(index = expression(localArray.index))
          .asInstanceOf[Place[T, Space, Mode]]

    private def rewriteChildren[T](expression: Expr[T]): Expr[T] = expression match
      case binary: Binary[?] =>
        binary
          .copy(
            left = this.expression(binary.left),
            right = this.expression(binary.right)
          )
          .asInstanceOf[Expr[T]]
      case comparison: Compare[?] =>
        comparison
          .copy(
            left = this.expression(comparison.left),
            right = this.expression(comparison.right)
          )
          .asInstanceOf[Expr[T]]
      case math: UnaryMath[?] =>
        math.copy(value = this.expression(math.value)).asInstanceOf[Expr[T]]
      case conversion: Convert[?, ?] =>
        conversion
          .copy(value = this.expression(conversion.value))
          .asInstanceOf[Expr[T]]
      case accumulation: ToAccumulator[?, ?] =>
        accumulation
          .copy(value = this.expression(accumulation.value))
          .asInstanceOf[Expr[T]]
      // Branch values must not be speculatively evaluated outside their condition.
      case _: Conditional[?] | _: ReduceSum[?, ?] => expression
      case load: Load[?, ?, ?] =>
        load.copy(from = place(load.from)).asInstanceOf[Expr[T]]
      case _: Literal[?] | _: ScalarParam[?] | _: Intrinsic[?] |
          _: ReductionIndex | _: LoopIndex =>
        expression

  private def rewriteAtomicFetch[T, Space <: AddressSpace](
      atomic: AtomicFetchAdd[T, Space],
      rewriter: ExpressionRewriter
  ): AtomicFetchAdd[T, Space] =
    atomic.copy(
      target = rewriter.place(atomic.target),
      value = rewriter.expression(atomic.value)
    )

  private def rewriteAtomic[T, Space <: AddressSpace](
      atomic: AtomicAdd[T, Space],
      rewriter: ExpressionRewriter
  ): AtomicAdd[T, Space] =
    atomic.copy(
      target = rewriter.place(atomic.target),
      value = rewriter.expression(atomic.value)
    )

  private def rewriteStore[T, Space <: AddressSpace](
      store: Store[T, Space],
      rewriter: ExpressionRewriter
  ): Store[T, Space] =
    store.copy(
      to = rewriter.place(store.to),
      value = rewriter.expression(store.value)
    )

  private def ownExpressions(statement: Stmt): Vector[Expr[?]] = statement match
    case declaration: LocalDeclaration[?] => Vector(declaration.initial)
    case _: LocalArrayDeclaration[?] => Vector.empty
    case vote: WarpVote[?] => Vector(vote.mask, vote.predicate)
    case store: Store[?, ?] => placeExpressions(store.to) :+ store.value
    case atomic: AtomicAdd[?, ?] => placeExpressions(atomic.target) :+ atomic.value
    case atomic: AtomicFetchAdd[?, ?] => placeExpressions(atomic.target) :+ atomic.value
    case accumulation: Accumulate[?] => Vector(accumulation.value)
    case branch: IfThen => Vector(branch.condition)
    case _: ScopedBlock => Vector.empty
    case loop: ForLoop => Vector(loop.from, loop.until)
    case _: Barrier => Vector.empty

  private def placeExpressions(place: Place[?, ?, ?]): Vector[Expr[?]] = place match
    case buffer: BufferElement[?, ?] => Vector(buffer.index)
    case constant: ConstantElement[?] => Vector(constant.index)
    case shared: SharedElement[?] => shared.indices
    case _: LocalVariable[?] => Vector.empty
    case localArray: LocalArrayElement[?] => Vector(localArray.index)

  private def collectCounts(
      expression: Expr[?],
      counts: mutable.Map[IntegerExpressionKey, Int]
  ): Unit =
    candidateKey(expression).foreach { key =>
      counts.update(key, counts.getOrElse(key, 0) + 1)
    }
    expression match
      case binary: Binary[?] =>
        collectCounts(binary.left, counts)
        collectCounts(binary.right, counts)
      case comparison: Compare[?] =>
        collectCounts(comparison.left, counts)
        collectCounts(comparison.right, counts)
      case math: UnaryMath[?] => collectCounts(math.value, counts)
      case conversion: Convert[?, ?] => collectCounts(conversion.value, counts)
      case accumulation: ToAccumulator[?, ?] => collectCounts(accumulation.value, counts)
      case _: Conditional[?] | _: ReduceSum[?, ?] => ()
      case load: Load[?, ?, ?] =>
        placeExpressions(load.from).foreach(collectCounts(_, counts))
      case _: Literal[?] | _: ScalarParam[?] | _: Intrinsic[?] |
          _: ReductionIndex | _: LoopIndex => ()

  private def candidateKey(expression: Expr[?]): Option[IntegerExpressionKey] =
    expression match
      case binary: Binary[?]
          if binary.valueType == I32 && EffectAnalysis.expression(binary).isPure =>
        integerKey(binary)
      case _ => None

  private def integerKey(expression: Expr[?]): Option[IntegerExpressionKey] =
    if expression.valueType != I32 then None
    else
      expression match
        case literal: Literal[?] =>
          literal.value match
            case value: Int => Some(IntegerExpressionKey.Literal(value))
            case _ => None
        case scalar: ScalarParam[?] =>
          Some(IntegerExpressionKey.Scalar(scalar.name))
        case intrinsic: Intrinsic[?] =>
          Some(IntegerExpressionKey.Intrinsic(intrinsic.name))
        case index: LoopIndex =>
          Some(IntegerExpressionKey.LoopIndex(index.name))
        case index: ReductionIndex =>
          Some(IntegerExpressionKey.ReductionIndex(index.name))
        case binary: Binary[?] =>
          for
            left <- integerKey(binary.left)
            right <- integerKey(binary.right)
          yield IntegerExpressionKey.Binary(binary.operator, left, right)
        case _: Compare[?] | _: Conditional[?] | _: UnaryMath[?] | _: Convert[?, ?] | _: ToAccumulator[?, ?] |
            _: ReduceSum[?, ?] | _: Load[?, ?, ?] => None

  private def collectNames(kernel: KernelIR[?]): Set[String] =
    kernel.params.map(_.name).toSet ++
      kernel.sharedMemory.map(_.name) ++
      collectNames(kernel.body) +
      kernel.name

  private def collectNames(block: Block): Set[String] =
    block.statements.flatMap(collectNames).toSet

  private def collectNames(statement: Stmt): Set[String] = statement match
    case declaration: LocalDeclaration[?] =>
      collectNames(declaration.initial) + declaration.local.name
    case declaration: LocalArrayDeclaration[?] => Set(declaration.array.name)
    case vote: WarpVote[?] => collectNames(vote.mask) ++ collectNames(vote.predicate) + vote.local.name
    case store: Store[?, ?] => collectNames(store.to) ++ collectNames(store.value)
    case atomic: AtomicAdd[?, ?] => collectNames(atomic.target) ++ collectNames(atomic.value)
    case atomic: AtomicFetchAdd[?, ?] =>
      collectNames(atomic.target) ++ collectNames(atomic.value) + atomic.local.name
    case accumulation: Accumulate[?] =>
      collectNames(accumulation.value) + accumulation.target.name
    case branch: IfThen =>
      collectNames(branch.condition) ++
        collectNames(branch.thenBlock) ++
        branch.elseBlock.toSet.flatMap(collectNames)
    case scoped: ScopedBlock => collectNames(scoped.body)
    case loop: ForLoop =>
      collectNames(loop.from) ++
        collectNames(loop.until) ++
        collectNames(loop.body) +
        loop.index.name
    case _: Barrier => Set.empty

  private def collectNames(expression: Expr[?]): Set[String] = expression match
    case _: Literal[?] => Set.empty
    case scalar: ScalarParam[?] => Set(scalar.name)
    case binary: Binary[?] => collectNames(binary.left) ++ collectNames(binary.right)
    case comparison: Compare[?] =>
      collectNames(comparison.left) ++ collectNames(comparison.right)
    case conditional: Conditional[?] =>
      collectNames(conditional.condition) ++
        collectNames(conditional.whenTrue) ++ collectNames(conditional.whenFalse)
    case math: UnaryMath[?] => collectNames(math.value)
    case intrinsic: Intrinsic[?] => Set(intrinsic.name)
    case conversion: Convert[?, ?] => collectNames(conversion.value)
    case accumulation: ToAccumulator[?, ?] => collectNames(accumulation.value)
    case index: ReductionIndex => Set(index.name)
    case index: LoopIndex => Set(index.name)
    case reduction: ReduceSum[?, ?] =>
      collectNames(reduction.from) ++
        collectNames(reduction.until) ++
        collectNames(reduction.initial) ++
        collectNames(reduction.value) +
        reduction.index.name
    case load: Load[?, ?, ?] => collectNames(load.from)

  private def collectNames(place: Place[?, ?, ?]): Set[String] = place match
    case buffer: BufferElement[?, ?] => collectNames(buffer.index) + buffer.bufferName
    case constant: ConstantElement[?] => collectNames(constant.index) + constant.arrayName
    case shared: SharedElement[?] =>
      shared.indices.flatMap(collectNames).toSet + shared.arrayName
    case local: LocalVariable[?] => Set(local.name)
    case localArray: LocalArrayElement[?] =>
      collectNames(localArray.index) + localArray.arrayName
