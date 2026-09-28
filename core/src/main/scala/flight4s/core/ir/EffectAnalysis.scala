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
    hasWarpCollective: Boolean = false,
    hasWarpBarrier: Boolean = false
):
  def isPure: Boolean =
    readSpaces.isEmpty && writtenSpaces.isEmpty && !hasBarrier && !hasWarpCollective && !hasWarpBarrier

  def ++(other: EffectSummary): EffectSummary =
    EffectSummary(
      readSpaces ++ other.readSpaces,
      writtenSpaces ++ other.writtenSpaces,
      hasBarrier || other.hasBarrier,
      hasWarpCollective || other.hasWarpCollective,
      hasWarpBarrier || other.hasWarpBarrier
    )

private[core] object EffectSummary:
  val empty: EffectSummary = EffectSummary()

private[core] object EffectAnalysis:
  def expression(expr: Expr[?]): EffectSummary = expr match
    case _: Literal[?] => EffectSummary.empty
    case vector: FloatVectorConstruct[?] =>
      vector.components.foldLeft(EffectSummary.empty)((effects, component) => effects ++ expression(component))
    case component: FloatVectorComponent[?] => expression(component.value)
    case binary: Binary[?] =>
      expression(binary.left) ++ expression(binary.right)
    case shift: UnsignedShift => expression(shift.value) ++ expression(shift.distance)
    case shift: SignedShift => expression(shift.value) ++ expression(shift.distance)
    case count: PopulationCount[?] => expression(count.value)
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
    case barrier: WarpBarrier =>
      expression(barrier.mask) ++ EffectSummary(hasWarpCollective = true, hasWarpBarrier = true)

  def block(block: Block): EffectSummary =
    block.statements.foldLeft(EffectSummary.empty) { (summary, next) =>
      summary ++ statement(next)
    }

  def modifiedLocalNames(block: Block): Set[String] =
    block.statements.flatMap(modifiedLocalNames).toSet

  private def modifiedLocalNames(statement: Stmt): Set[String] = statement match
    case store: Store[?, ?] =>
      store.to match
        case local: LocalVariable[?] => Set(local.name)
        case _ => Set.empty
    case accumulation: Accumulate[?] => Set(accumulation.target.name)
    case branch: IfThen =>
      modifiedLocalNames(branch.thenBlock) ++
        branch.elseBlock.toVector.flatMap(modifiedLocalNames).toSet
    case scoped: ScopedBlock => modifiedLocalNames(scoped.body)
    case loop: ForLoop => modifiedLocalNames(loop.body)
    case _: LocalDeclaration[?] | _: LocalArrayDeclaration[?] |
        _: AtomicAdd[?, ?] | _: AtomicFetchAdd[?, ?] | _: WarpVote[?] | _: WarpShuffle[?, ?] |
        _: Barrier | _: WarpBarrier => Set.empty

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
