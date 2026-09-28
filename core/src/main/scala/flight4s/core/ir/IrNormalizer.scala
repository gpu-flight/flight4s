package flight4s.core.ir

import flight4s.core.types.{Bool, I32, U32, UInt}

private[core] object IrNormalizer:
  private final case class ConstantScope(
      integerLocals: Map[String, Int] = Map.empty
  ):
    def bind(local: LocalVariable[?], value: Expr[?]): ConstantScope =
      if local.valueType != I32 then without(local.name)
      else
        integerLiteralValue(value) match
          case Some(integer) => copy(integerLocals = integerLocals.updated(local.name, integer))
          case None => without(local.name)

    def resolve(local: LocalVariable[?], span: SourceSpan): Option[Literal[Int]] =
      if local.valueType != I32 then None
      else integerLocals.get(local.name).map(value => Literal(value, I32, span))

    def without(name: String): ConstantScope =
      copy(integerLocals = integerLocals.removed(name))

  private object ConstantScope:
    val empty: ConstantScope = ConstantScope()

  def module(module: CudaModuleIR): CudaModuleIR =
    val reservedNames = module.constants.map(_.name).toSet
    module.copy(
      kernels = module.kernels.map(normalizeKernel(_, reservedNames))
    )

  def kernel[Args <: Tuple](kernel: KernelIR[Args]): KernelIR[Args] =
    normalizeKernel(kernel, Set.empty)

  def block(block: Block): Block =
    LocalCommonSubexpressionElimination.block(
      normalizeBlock(block, ConstantScope.empty)._1
    )

  def statement(statement: Stmt): Vector[Stmt] =
    normalizeStatement(statement, ConstantScope.empty)._1.toVector.flatMap { normalized =>
      LocalCommonSubexpressionElimination.block(Block(Vector(normalized))).statements
    }

  def expression[T](expr: Expr[T]): Expr[T] =
    expression(expr, ConstantScope.empty)

  private def normalizeKernel[Args <: Tuple](
      kernel: KernelIR[Args],
      reservedNames: Set[String]
  ): KernelIR[Args] =
    val normalized = kernel.copy(
      body = normalizeBlock(kernel.body, ConstantScope.empty)._1
    )
    LocalCommonSubexpressionElimination.kernel(normalized, reservedNames)

  private def normalizeBlock(
      block: Block,
      initialScope: ConstantScope
  ): (Block, ConstantScope) =
    val (statements, finalScope) = block.statements.foldLeft(
      (Vector.empty[Stmt], initialScope)
    ) { case ((normalizedStatements, scope), statement) =>
      val (normalizedStatement, nextScope) = normalizeStatement(statement, scope)
      (normalizedStatements ++ normalizedStatement, nextScope)
    }
    (Block(statements), finalScope)

  private def normalizeStatement(
      statement: Stmt,
      scope: ConstantScope
  ): (Option[Stmt], ConstantScope) = statement match
    case declaration: LocalDeclaration[?] =>
      val normalized = declaration.copy(
        initial = expression(declaration.initial, scope)
      )
      (Some(normalized), scope.bind(declaration.local, normalized.initial))

    case declaration: LocalArrayDeclaration[?] => (Some(declaration), scope)

    case shuffle: WarpShuffle[?, ?] =>
      (Some(shuffle.copy(mask = expression(shuffle.mask, scope), value = expression(shuffle.value, scope),
        selector = expression(shuffle.selector, scope))), scope.without(shuffle.local.name))

    case vote: WarpVote[?] =>
      (Some(vote.copy(mask = expression(vote.mask, scope), predicate = expression(vote.predicate, scope))),
        scope.without(vote.local.name))

    case store: Store[?, ?] =>
      val normalized = normalizeStore(store, scope)
      val nextScope = normalized.to match
        case local: LocalVariable[?] => scope.bind(local, normalized.value)
        case _ => scope
      (Some(normalized), nextScope)

    case atomic: AtomicAdd[?, ?] =>
      (Some(normalizeAtomic(atomic, scope)), scope)

    case atomic: AtomicFetchAdd[?, ?] =>
      (Some(normalizeAtomicFetch(atomic, scope)), scope.without(atomic.local.name))

    case accumulation: Accumulate[?] =>
      val normalized = accumulation.copy(value = expression(accumulation.value, scope))
      (Some(normalized), scope.without(accumulation.target.name))

    case branch: IfThen =>
      val condition = expression(branch.condition, scope)
      booleanLiteralValue(condition) match
        case Some(true) =>
          normalizeSelectedBlock(branch.thenBlock, branch.span, scope)
        case Some(false) =>
          branch.elseBlock match
            case Some(elseBlock) =>
              normalizeSelectedBlock(elseBlock, branch.span, scope)
            case None => (None, scope)
        case None =>
          val (thenBlock, _) = normalizeBlock(branch.thenBlock, scope)
          val elseBlock = branch.elseBlock.map { block =>
            normalizeBlock(block, scope)._1
          }
          (
            Some(
              branch.copy(
                condition = condition,
                thenBlock = thenBlock,
                elseBlock = elseBlock
              )
            ),
            ConstantScope.empty
          )

    case scoped: ScopedBlock =>
      normalizeSelectedBlock(scoped.body, scoped.span, scope)

    case loop: ForLoop =>
      val (body, _) = normalizeBlock(loop.body, ConstantScope.empty)
      (
        Some(
          loop.copy(
            from = expression(loop.from, scope),
            until = expression(loop.until, scope),
            body = body
          )
        ),
        ConstantScope.empty
      )

    case barrier: Barrier => (Some(barrier), scope)
    case barrier: WarpBarrier => (Some(barrier.copy(mask = expression(barrier.mask, scope))), scope)

  private def normalizeSelectedBlock(
      block: Block,
      span: SourceSpan,
      scope: ConstantScope
  ): (Option[Stmt], ConstantScope) =
    val (body, _) = normalizeBlock(block, scope)
    if body.statements.isEmpty then (None, scope)
    else (Some(ScopedBlock(body, span)), ConstantScope.empty)

  private def expression[T](expr: Expr[T], scope: ConstantScope): Expr[T] = expr match
    case literal: Literal[?] => literal.asInstanceOf[Expr[T]]
    case binary: Binary[?] =>
      normalizeBinary(binary, scope).asInstanceOf[Expr[T]]
    case shift: UnsignedShift => normalizeUnsignedShift(shift, scope).asInstanceOf[Expr[T]]
    case shift: SignedShift => normalizeSignedShift(shift, scope).asInstanceOf[Expr[T]]
    case count: PopulationCount[?] => normalizePopulationCount(count, scope).asInstanceOf[Expr[T]]
    case comparison: Compare[?] =>
      normalizeComparison(comparison, scope).asInstanceOf[Expr[T]]
    case conditional: Conditional[?] =>
      normalizeConditional(conditional, scope).asInstanceOf[Expr[T]]
    case math: UnaryMath[?] =>
      math.copy(value = expression(math.value, scope)).asInstanceOf[Expr[T]]
    case intrinsic: Intrinsic[?] => intrinsic.asInstanceOf[Expr[T]]
    case conversion: Convert[?, ?] =>
      conversion
        .copy(value = expression(conversion.value, scope))
        .asInstanceOf[Expr[T]]
    case accumulation: ToAccumulator[?, ?] =>
      accumulation
        .copy(value = expression(accumulation.value, scope))
        .asInstanceOf[Expr[T]]
    case index: ReductionIndex => index.asInstanceOf[Expr[T]]
    case index: LoopIndex => index.asInstanceOf[Expr[T]]
    case reduction: ReduceSum[?, ?] =>
      reduction
        .copy(
          from = expression(reduction.from, scope),
          until = expression(reduction.until, scope),
          initial = expression(reduction.initial, scope),
          value = expression(reduction.value, scope)
        )
        .asInstanceOf[Expr[T]]
    case load: Load[?, ?, ?] =>
      normalizeLoad(load, scope).asInstanceOf[Expr[T]]
    case parameter: ScalarParam[?] => parameter.asInstanceOf[Expr[T]]

  private def normalizeConditional[T](
      conditional: Conditional[T],
      scope: ConstantScope
  ): Expr[T] =
    val condition = expression(conditional.condition, scope)
    booleanLiteralValue(condition) match
      case Some(true) => withSpan(expression(conditional.whenTrue, scope), conditional.span)
      case Some(false) => withSpan(expression(conditional.whenFalse, scope), conditional.span)
      case None => conditional.copy(
        condition = condition,
        whenTrue = expression(conditional.whenTrue, scope),
        whenFalse = expression(conditional.whenFalse, scope)
      )

  private def normalizeAtomicFetch[T, Space <: AddressSpace](
      atomic: AtomicFetchAdd[T, Space],
      scope: ConstantScope
  ): AtomicFetchAdd[T, Space] =
    atomic.copy(
      target = place(atomic.target, scope),
      value = expression(atomic.value, scope)
    )

  private def normalizeAtomic[T, Space <: AddressSpace](
      atomic: AtomicAdd[T, Space],
      scope: ConstantScope
  ): AtomicAdd[T, Space] =
    atomic.copy(
      target = place(atomic.target, scope),
      value = expression(atomic.value, scope)
    )

  private def normalizeStore[T, Space <: AddressSpace](
      store: Store[T, Space],
      scope: ConstantScope
  ): Store[T, Space] =
    store.copy(
      to = place(store.to, scope),
      value = expression(store.value, scope)
    )

  private def normalizeLoad[
      T,
      Space <: AddressSpace,
      Mode <: AccessMode
  ](
      load: Load[T, Space, Mode],
      scope: ConstantScope
  ): Expr[T] = load.from match
    case local: LocalVariable[?] =>
      scope.resolve(local, load.span) match
        case Some(literal) => literal.asInstanceOf[Expr[T]]
        case None => load.copy(from = place(load.from, scope))
    case _ => load.copy(from = place(load.from, scope))

  private def normalizeBinary[T](
      binary: Binary[T],
      scope: ConstantScope
  ): Expr[T] =
    val left = expression(binary.left, scope)
    val right = expression(binary.right, scope)
    foldIntegerBinary(binary, left, right)
      .orElse(simplifyIntegerIdentity(binary, left, right))
      .getOrElse(binary.copy(left = left, right = right))

  private def normalizeUnsignedShift(shift: UnsignedShift, scope: ConstantScope): Expr[UInt] =
    val value = expression(shift.value, scope)
    val distance = expression(shift.distance, scope)
    val folded = for
      bits <- unsignedLiteralBits(value)
      count <- integerLiteralValue(distance)
    yield
      val result = shift.operator match
        case UnsignedShiftOperator.Left => bits << (count & 31)
        case UnsignedShiftOperator.Right => bits >>> (count & 31)
      Literal(UInt.fromBits(result), U32, shift.span)
    folded.getOrElse(shift.copy(value = value, distance = distance))

  private def normalizeSignedShift(shift: SignedShift, scope: ConstantScope): Expr[Int] =
    val value = expression(shift.value, scope)
    val distance = expression(shift.distance, scope)
    val folded = for
      bits <- integerLiteralValue(value)
      count <- integerLiteralValue(distance)
    yield
      val result = shift.operator match
        case SignedShiftOperator.Left => bits << (count & 31)
        case SignedShiftOperator.ArithmeticRight => bits >> (count & 31)
        case SignedShiftOperator.LogicalRight => bits >>> (count & 31)
      Literal(result, I32, shift.span)
    folded.getOrElse(shift.copy(value = value, distance = distance))

  private def normalizePopulationCount[T](count: PopulationCount[T], scope: ConstantScope): Expr[Int] =
    val value = expression(count.value, scope)
    integerLiteralValue(value).orElse(unsignedLiteralBits(value))
      .map(word => Literal(java.lang.Integer.bitCount(word), I32, count.span))
      .getOrElse(count.copy(value = value))

  private def unsignedLiteralBits(expression: Expr[?]): Option[Int] = expression match
    case literal: Literal[?] if literal.valueType == U32 => literal.value match
      case value: UInt => Some(value.toIntBits)
      case _ => None
    case _ => None

  private def normalizeComparison[T](
      comparison: Compare[T],
      scope: ConstantScope
  ): Expr[Boolean] =
    val left = expression(comparison.left, scope)
    val right = expression(comparison.right, scope)
    foldIntegerComparison(comparison, left, right).getOrElse(
      comparison.copy(left = left, right = right)
    )

  private def place[
      T,
      Space <: AddressSpace,
      Mode <: AccessMode
  ](
      place: Place[T, Space, Mode],
      scope: ConstantScope
  ): Place[T, Space, Mode] = place match
    case buffer: BufferElement[?, ?] =>
      buffer
        .copy(index = expression(buffer.index, scope))
        .asInstanceOf[Place[T, Space, Mode]]
    case constant: ConstantElement[?] =>
      constant
        .copy(index = expression(constant.index, scope))
        .asInstanceOf[Place[T, Space, Mode]]
    case shared: SharedElement[?] =>
      shared
        .copy(indices = shared.indices.map(expression(_, scope)))
        .asInstanceOf[Place[T, Space, Mode]]
    case local: LocalVariable[?] => local.asInstanceOf[Place[T, Space, Mode]]
    case localArray: LocalArrayElement[?] =>
      localArray
        .copy(index = expression(localArray.index, scope))
        .asInstanceOf[Place[T, Space, Mode]]

  private def foldIntegerBinary[T](
      binary: Binary[T],
      left: Expr[T],
      right: Expr[T]
  ): Option[Expr[T]] =
    if binary.valueType != I32 then None
    else
      (left, right) match
        case (leftLiteral: Literal[?], rightLiteral: Literal[?]) =>
          (leftLiteral.value, rightLiteral.value) match
            case (leftValue: Int, rightValue: Int) =>
              integerBinaryResult(binary.operator, leftValue, rightValue).map { value =>
                Literal(value, I32, binary.span).asInstanceOf[Expr[T]]
              }
            case _ => None
        case _ => None

  private def simplifyIntegerIdentity[T](
      binary: Binary[T],
      left: Expr[T],
      right: Expr[T]
  ): Option[Expr[T]] =
    if binary.valueType != I32 then None
    else
      (binary.operator, integerLiteralValue(left), integerLiteralValue(right)) match
        case (BinaryOperator.Add, _, Some(0)) =>
          Some(withSpan(left, binary.span))
        case (BinaryOperator.Add, Some(0), _) =>
          Some(withSpan(right, binary.span))
        case (BinaryOperator.Subtract, _, Some(0)) =>
          Some(withSpan(left, binary.span))
        case (BinaryOperator.Multiply, _, Some(1)) =>
          Some(withSpan(left, binary.span))
        case (BinaryOperator.Multiply, Some(1), _) =>
          Some(withSpan(right, binary.span))
        case (BinaryOperator.Divide, _, Some(1)) =>
          Some(withSpan(left, binary.span))
        case _ => None

  // Rewrites retain the complete source expression for generated-CUDA diagnostics.
  private def withSpan[T](expression: Expr[T], span: SourceSpan): Expr[T] =
    expression match
      case literal: Literal[?] => literal.copy(span = span).asInstanceOf[Expr[T]]
      case binary: Binary[?] => binary.copy(span = span).asInstanceOf[Expr[T]]
      case shift: UnsignedShift => shift.copy(span = span).asInstanceOf[Expr[T]]
      case shift: SignedShift => shift.copy(span = span).asInstanceOf[Expr[T]]
      case count: PopulationCount[?] => count.copy(span = span).asInstanceOf[Expr[T]]
      case comparison: Compare[?] => comparison.copy(span = span).asInstanceOf[Expr[T]]
      case conditional: Conditional[?] => conditional.copy(span = span).asInstanceOf[Expr[T]]
      case math: UnaryMath[?] => math.copy(span = span).asInstanceOf[Expr[T]]
      case intrinsic: Intrinsic[?] => intrinsic.copy(span = span).asInstanceOf[Expr[T]]
      case conversion: Convert[?, ?] => conversion.copy(span = span).asInstanceOf[Expr[T]]
      case accumulation: ToAccumulator[?, ?] =>
        accumulation.copy(span = span).asInstanceOf[Expr[T]]
      case index: ReductionIndex => index.copy(span = span).asInstanceOf[Expr[T]]
      case index: LoopIndex => index.copy(span = span).asInstanceOf[Expr[T]]
      case reduction: ReduceSum[?, ?] => reduction.copy(span = span).asInstanceOf[Expr[T]]
      case load: Load[?, ?, ?] => load.copy(span = span).asInstanceOf[Expr[T]]
      case parameter: ScalarParam[?] =>
        parameter.copy(span = span)(using parameter.scalarAbi).asInstanceOf[Expr[T]]

  private def integerBinaryResult(
      operator: BinaryOperator,
      left: Int,
      right: Int
  ): Option[Int] =
    operator match
      case BinaryOperator.Add =>
        try Some(Math.addExact(left, right))
        catch case _: ArithmeticException => None
      case BinaryOperator.Subtract =>
        try Some(Math.subtractExact(left, right))
        catch case _: ArithmeticException => None
      case BinaryOperator.Multiply =>
        try Some(Math.multiplyExact(left, right))
        catch case _: ArithmeticException => None
      case BinaryOperator.Divide if right == 0 || (left == Int.MinValue && right == -1) =>
        None
      case BinaryOperator.Divide => Some(left / right)
      case BinaryOperator.Remainder
          if right == 0 || (left == Int.MinValue && right == -1) =>
        None
      case BinaryOperator.Remainder => Some(left % right)
      case BinaryOperator.BitAnd => Some(left & right)
      case BinaryOperator.BitOr => Some(left | right)
      case BinaryOperator.BitXor => Some(left ^ right)

  private def foldIntegerComparison[T](
      comparison: Compare[T],
      left: Expr[T],
      right: Expr[T]
  ): Option[Expr[Boolean]] =
    if comparison.operandType != I32 then None
    else
      (left, right) match
        case (leftLiteral: Literal[?], rightLiteral: Literal[?]) =>
          (leftLiteral.value, rightLiteral.value) match
            case (leftValue: Int, rightValue: Int) =>
              Some(
                Literal(
                  integerComparisonResult(comparison.operator, leftValue, rightValue),
                  Bool,
                  comparison.span
                )
              )
            case _ => None
        case _ => None

  private def integerComparisonResult(
      operator: ComparisonOperator,
      left: Int,
      right: Int
  ): Boolean =
    operator match
      case ComparisonOperator.LessThan => left < right
      case ComparisonOperator.LessThanOrEqual => left <= right
      case ComparisonOperator.GreaterThan => left > right
      case ComparisonOperator.GreaterThanOrEqual => left >= right
      case ComparisonOperator.Equal => left == right
      case ComparisonOperator.NotEqual => left != right

  private def integerLiteralValue(expression: Expr[?]): Option[Int] = expression match
    case literal: Literal[?] if literal.valueType == I32 =>
      literal.value match
        case value: Int => Some(value)
        case _ => None
    case _ => None

  private def booleanLiteralValue(expression: Expr[?]): Option[Boolean] =
    expression match
      case literal: Literal[?] if literal.valueType == Bool =>
        literal.value match
          case value: Boolean => Some(value)
          case _ => None
      case _ => None
