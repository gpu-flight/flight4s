package flight4s.core.ir

private[core] enum EffectMemorySpace:
  case Global
  case Shared
  case Local
  case Constant

private[core] final case class EffectSummary(
    readSpaces: Set[EffectMemorySpace] = Set.empty,
    writtenSpaces: Set[EffectMemorySpace] = Set.empty,
    hasBarrier: Boolean = false,
    hasWarpCollective: Boolean = false
):
  def isPure: Boolean =
    readSpaces.isEmpty && writtenSpaces.isEmpty && !hasBarrier && !hasWarpCollective

  def ++(other: EffectSummary): EffectSummary =
    EffectSummary(
      readSpaces ++ other.readSpaces,
      writtenSpaces ++ other.writtenSpaces,
      hasBarrier || other.hasBarrier,
      hasWarpCollective || other.hasWarpCollective
    )

private[core] object EffectSummary:
  val empty: EffectSummary = EffectSummary()

private[core] object EffectAnalysis:
  def expression(expr: Expr[?]): EffectSummary = expr match
    case _: Literal[?] => EffectSummary.empty
    case binary: Binary[?] =>
      expression(binary.left) ++ expression(binary.right)
    case comparison: Compare[?] =>
      expression(comparison.left) ++ expression(comparison.right)
    case conditional: Conditional[?] =>
      expression(conditional.condition) ++
        expression(conditional.whenTrue) ++ expression(conditional.whenFalse)
    case math: UnaryMath[?] => expression(math.value)
    case _: Intrinsic[?] => EffectSummary.empty
    case conversion: Convert[?, ?] => expression(conversion.value)
    case accumulation: ToAccumulator[?, ?] => expression(accumulation.value)
    case _: ReductionIndex => EffectSummary.empty
    case _: LoopIndex => EffectSummary.empty
    case reduction: ReduceSum[?, ?] =>
      expression(reduction.from) ++
        expression(reduction.until) ++
        expression(reduction.initial) ++
        expression(reduction.value)
    case load: Load[?, ?, ?] => read(load.from)
    case _: ScalarParam[?] => EffectSummary.empty

  def statement(statement: Stmt): EffectSummary = statement match
    case declaration: LocalDeclaration[?] =>
      expression(declaration.initial) ++ write(EffectMemorySpace.Local)
    case _: LocalArrayDeclaration[?] => EffectSummary.empty
    case vote: WarpVote[?] =>
      expression(vote.mask) ++ expression(vote.predicate) ++
        write(EffectMemorySpace.Local) ++ EffectSummary(hasWarpCollective = true)
    case shuffle: WarpShuffle[?, ?] =>
      expression(shuffle.mask) ++ expression(shuffle.value) ++ expression(shuffle.selector) ++
        write(EffectMemorySpace.Local) ++ EffectSummary(hasWarpCollective = true)
    case store: Store[?, ?] =>
      addressEffects(store.to) ++
        expression(store.value) ++
        write(spaceOf(store.to))
    case atomic: AtomicAdd[?, ?] =>
      read(atomic.target) ++ expression(atomic.value) ++ write(spaceOf(atomic.target))
    case atomic: AtomicFetchAdd[?, ?] =>
      read(atomic.target) ++ expression(atomic.value) ++
        write(spaceOf(atomic.target)) ++ write(EffectMemorySpace.Local)
    case accumulation: Accumulate[?] =>
      expression(accumulation.value) ++
        read(EffectMemorySpace.Local) ++
        write(EffectMemorySpace.Local)
    case branch: IfThen =>
      expression(branch.condition) ++
        block(branch.thenBlock) ++
        branch.elseBlock.map(block).getOrElse(EffectSummary.empty)
    case scoped: ScopedBlock => block(scoped.body)
    case loop: ForLoop =>
      expression(loop.from) ++ expression(loop.until) ++ block(loop.body)
    case _: Barrier => EffectSummary(hasBarrier = true)

  def block(block: Block): EffectSummary =
    block.statements.foldLeft(EffectSummary.empty) { (summary, next) =>
      summary ++ statement(next)
    }

  private def read(place: Place[?, ?, ?]): EffectSummary =
    addressEffects(place) ++ EffectSummary(readSpaces = Set(spaceOf(place)))

  private def read(space: EffectMemorySpace): EffectSummary =
    EffectSummary(readSpaces = Set(space))

  private def write(space: EffectMemorySpace): EffectSummary =
    EffectSummary(writtenSpaces = Set(space))

  private def addressEffects(place: Place[?, ?, ?]): EffectSummary = place match
    case buffer: BufferElement[?, ?] => expression(buffer.index)
    case constant: ConstantElement[?] => expression(constant.index)
    case shared: SharedElement[?] =>
      shared.indices.foldLeft(EffectSummary.empty) { (summary, index) =>
        summary ++ expression(index)
      }
    case _: LocalVariable[?] => EffectSummary.empty
    case localArray: LocalArrayElement[?] => expression(localArray.index)

  private def spaceOf(place: Place[?, ?, ?]): EffectMemorySpace = place match
    case _: BufferElement[?, ?] => EffectMemorySpace.Global
    case _: ConstantElement[?] => EffectMemorySpace.Constant
    case _: SharedElement[?] => EffectMemorySpace.Shared
    case _: LocalVariable[?] => EffectMemorySpace.Local
    case _: LocalArrayElement[?] => EffectMemorySpace.Local
