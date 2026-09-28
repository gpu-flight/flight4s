package flight4s.core.ir

private[core] enum Uniformity:
  case GridUniform
  case BlockUniform
  case WarpUniform
  case Varying
  case Unknown

  def join(other: Uniformity): Uniformity = (this, other) match
    case (Uniformity.Unknown, _) | (_, Uniformity.Unknown) => Uniformity.Unknown
    case (Uniformity.Varying, _) | (_, Uniformity.Varying) => Uniformity.Varying
    case (Uniformity.WarpUniform, _) | (_, Uniformity.WarpUniform) =>
      Uniformity.WarpUniform
    case (Uniformity.BlockUniform, _) | (_, Uniformity.BlockUniform) =>
      Uniformity.BlockUniform
    case _ => Uniformity.GridUniform

private[core] final case class UniformityScope(
    locals: Map[String, Uniformity] = Map.empty,
    loopIndexes: Map[String, Uniformity] = Map.empty,
    reductionIndexes: Map[String, Uniformity] = Map.empty
):
  def withLocal(name: String, uniformity: Uniformity): UniformityScope =
    copy(locals = locals.updated(name, uniformity))

  def withLoopIndex(name: String, uniformity: Uniformity): UniformityScope =
    copy(loopIndexes = loopIndexes.updated(name, uniformity))

  def withReductionIndex(name: String, uniformity: Uniformity): UniformityScope =
    copy(reductionIndexes = reductionIndexes.updated(name, uniformity))

  def withVisibleLocalsFrom(other: UniformityScope): UniformityScope =
    copy(
      locals = locals.map { case (name, uniformity) =>
        name -> other.locals.getOrElse(name, uniformity)
      }
    )

  def mergeVisible(others: UniformityScope*): UniformityScope =
    copy(
      locals = locals.map { case (name, uniformity) =>
        name -> others.foldLeft(uniformity) { (merged, other) =>
          merged.join(other.locals.getOrElse(name, uniformity))
        }
      }
    )

  def withControlDependence(
      names: Set[String],
      uniformity: Uniformity
  ): UniformityScope =
    copy(
      locals = locals.map { case (name, current) =>
        name ->
          (if names.contains(name) then current.join(uniformity) else current)
      }
    )

private[core] object UniformityScope:
  val empty: UniformityScope = UniformityScope()

private[core] object UniformityAnalysis:
  import EffectAnalysis.modifiedLocalNames

  private val intrinsicUniformities: Map[String, Uniformity] = Map(
    "threadIdx.x" -> Uniformity.Varying,
    "threadIdx.y" -> Uniformity.Varying,
    "threadIdx.z" -> Uniformity.Varying,
    "blockIdx.x" -> Uniformity.BlockUniform,
    "blockIdx.y" -> Uniformity.BlockUniform,
    "blockIdx.z" -> Uniformity.BlockUniform,
    "blockDim.x" -> Uniformity.GridUniform,
    "blockDim.y" -> Uniformity.GridUniform,
    "blockDim.z" -> Uniformity.GridUniform,
    "gridDim.x" -> Uniformity.GridUniform,
    "gridDim.y" -> Uniformity.GridUniform,
    "gridDim.z" -> Uniformity.GridUniform
  )

  def expression(expr: Expr[?]): Uniformity =
    expression(expr, UniformityScope.empty)

  def expression(expr: Expr[?], scope: UniformityScope): Uniformity = expr match
    case _: Literal[?] => Uniformity.GridUniform
    case vector: FloatVectorConstruct[?] =>
      vector.components.foldLeft(Uniformity.GridUniform)((uniformity, component) => uniformity.join(expression(component, scope)))
    case component: FloatVectorComponent[?] => expression(component.value, scope)
    case binary: Binary[?] =>
      expression(binary.left, scope).join(expression(binary.right, scope))
    case shift: UnsignedShift =>
      expression(shift.value, scope).join(expression(shift.distance, scope))
    case shift: SignedShift =>
      expression(shift.value, scope).join(expression(shift.distance, scope))
    case count: PopulationCount[?] => expression(count.value, scope)
    case comparison: Compare[?] =>
      expression(comparison.left, scope).join(expression(comparison.right, scope))
    case conditional: Conditional[?] =>
      expression(conditional.condition, scope)
        .join(expression(conditional.whenTrue, scope))
        .join(expression(conditional.whenFalse, scope))
    case math: UnaryMath[?] => expression(math.value, scope)
    case intrinsic: Intrinsic[?] =>
      intrinsicUniformities.getOrElse(intrinsic.name, Uniformity.Unknown)
    case conversion: Convert[?, ?] => expression(conversion.value, scope)
    case accumulation: ToAccumulator[?, ?] => expression(accumulation.value, scope)
    case index: ReductionIndex =>
      scope.reductionIndexes.getOrElse(index.name, Uniformity.Unknown)
    case index: LoopIndex =>
      scope.loopIndexes.getOrElse(index.name, Uniformity.Unknown)
    case reduction: ReduceSum[?, ?] =>
      val indexUniformity =
        expression(reduction.from, scope).join(expression(reduction.until, scope))
      val reductionScope =
        scope.withReductionIndex(reduction.index.name, indexUniformity)
      indexUniformity
        .join(expression(reduction.initial, scope))
        .join(expression(reduction.value, reductionScope))
    case load: Load[?, ?, ?] => loadUniformity(load.from, scope)
    case _: ScalarParam[?] => Uniformity.GridUniform

  def scopeAfter(statement: Stmt, scope: UniformityScope): UniformityScope =
    statement match
      case declaration: LocalDeclaration[?] =>
        scope.withLocal(
          declaration.local.name,
          expression(declaration.initial, scope)
        )

      case store: Store[?, ?] =>
        store.to match
          case local: LocalVariable[?] =>
            scope.withLocal(local.name, expression(store.value, scope))
          case _ => scope

      case shuffle: WarpShuffle[?, ?] =>
        scope.withLocal(shuffle.local.name, Uniformity.Varying)

      case vote: WarpVote[?] =>
        // Disjoint participant groups in one physical warp can receive different results.
        scope.withLocal(vote.local.name, Uniformity.Varying)

      case atomic: AtomicFetchAdd[?, ?] =>
        scope.withLocal(atomic.local.name, Uniformity.Varying)

      case atomic: AtomicResult[?, ?] =>
        scope.withLocal(atomic.local.name, Uniformity.Varying)

      case accumulation: Accumulate[?] =>
        val current =
          scope.locals.getOrElse(accumulation.target.name, Uniformity.Unknown)
        scope.withLocal(
          accumulation.target.name,
          current.join(expression(accumulation.value, scope))
        )

      case branch: IfThen =>
        val thenScope = scopeAfter(branch.thenBlock, scope)
        val elseScope = branch.elseBlock match
          case Some(elseBlock) => scopeAfter(elseBlock, scope)
          case None => scope
        scope
          .mergeVisible(thenScope, elseScope)
          .withControlDependence(
            modifiedLocalNames(branch.thenBlock) ++
              branch.elseBlock.toVector.flatMap(modifiedLocalNames).toSet,
            expression(branch.condition, scope)
          )

      case scoped: ScopedBlock =>
        scope.withVisibleLocalsFrom(scopeAfter(scoped.body, scope))

      case loop: ForLoop =>
        val bodyScope = scopeAfter(loop.body, loopScope(loop, scope))
        scope
          .mergeVisible(bodyScope)
          .withControlDependence(
            modifiedLocalNames(loop.body),
            loopUniformity(loop, scope)
          )

      case _: LocalArrayDeclaration[?] | _: AtomicAdd[?, ?] | _: AtomicStore[?, ?] | _: Barrier | _: WarpBarrier => scope

  def scopeAfter(block: Block, scope: UniformityScope): UniformityScope =
    block.statements.foldLeft(scope) { (current, statement) =>
      scopeAfter(statement, current)
    }

  def loopScope(loop: ForLoop, scope: UniformityScope): UniformityScope =
    scope.withLoopIndex(loop.index.name, loopUniformity(loop, scope))

  private def loadUniformity(
      place: Place[?, ?, ?],
      scope: UniformityScope
  ): Uniformity = place match
    case constant: ConstantElement[?] => expression(constant.index, scope)
    case local: LocalVariable[?] =>
      scope.locals.getOrElse(local.name, Uniformity.Unknown)
    case _: BufferElement[?, ?] | _: SharedElement[?] | _: LocalArrayElement[?] =>
      Uniformity.Varying

  private def loopUniformity(
      loop: ForLoop,
      scope: UniformityScope
  ): Uniformity =
    expression(loop.from, scope).join(expression(loop.until, scope))
