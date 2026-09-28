package flight4s.core.ir

private[core] object BarrierDivergenceAnalysis:
  def warnings(block: Block): Vector[ValidationWarning] =
    blockWarnings(
      block,
      "body",
      controlMayDiverge = false,
      warpControlMayDiverge = false,
      UniformityScope.empty
    )

  private def blockWarnings(
      block: Block,
      location: String,
      controlMayDiverge: Boolean,
      warpControlMayDiverge: Boolean,
      initialScope: UniformityScope
  ): Vector[ValidationWarning] =
    block.statements.zipWithIndex
      .foldLeft((Vector.empty[ValidationWarning], initialScope)) {
        case ((warnings, scope), (statement, index)) =>
          val statementWarnings = warningsForStatement(
            statement,
            s"$location.statements[$index]",
            controlMayDiverge,
            warpControlMayDiverge,
            scope
          )
          (
            warnings ++ statementWarnings,
            UniformityAnalysis.scopeAfter(statement, scope)
          )
      }
      ._1

  private def warningsForStatement(
      statement: Stmt,
      statementLocation: String,
      controlMayDiverge: Boolean,
      warpControlMayDiverge: Boolean,
      scope: UniformityScope
  ): Vector[ValidationWarning] =
    statement match
      case branch: IfThen =>
        val branchMayDiverge =
          controlMayDiverge || isKnownDivergent(branch.condition, scope)
        val warpBranchMayDiverge =
          warpControlMayDiverge || isLaneVarying(branch.condition, scope)
        blockWarnings(branch.thenBlock, s"$statementLocation.then", branchMayDiverge, warpBranchMayDiverge, scope) ++
          branch.elseBlock.toVector.flatMap(
            blockWarnings(_, s"$statementLocation.else", branchMayDiverge, warpBranchMayDiverge, scope)
          )

      case scoped: ScopedBlock =>
        blockWarnings(
          scoped.body,
          s"$statementLocation.body",
          controlMayDiverge,
          warpControlMayDiverge,
          scope
        )

      case loop: ForLoop =>
        val loopMayDiverge =
          controlMayDiverge ||
            isKnownDivergent(loop.from, scope) ||
            isKnownDivergent(loop.until, scope)
        val warpLoopMayDiverge =
          warpControlMayDiverge || isLaneVarying(loop.from, scope) || isLaneVarying(loop.until, scope)
        blockWarnings(
          loop.body,
          s"$statementLocation.body",
          loopMayDiverge,
          warpLoopMayDiverge,
          UniformityAnalysis.loopScope(loop, scope)
        )

      case barrier: Barrier if controlMayDiverge =>
        Vector(
          ValidationWarning(
            ValidationWarningCode.BarrierMayDiverge,
            "block barrier may be reached through divergent control flow",
            statementLocation,
            barrier.span
          )
        )

      case _: Barrier => Vector.empty
      case barrier: WarpBarrier =>
        warpWarnings(barrier.mask, barrier.span, statementLocation, warpControlMayDiverge, scope)
      case _: LocalDeclaration[?] => Vector.empty
      case _: LocalArrayDeclaration[?] => Vector.empty
      case _: Store[?, ?] => Vector.empty
      case _: AtomicAdd[?, ?] => Vector.empty
      case _: AtomicFetchAdd[?, ?] => Vector.empty
      case vote: WarpVote[?] =>
        warpWarnings(vote.mask, vote.span, statementLocation, warpControlMayDiverge, scope)
      case shuffle: WarpShuffle[?, ?] =>
        warpWarnings(shuffle.mask, shuffle.span, statementLocation, warpControlMayDiverge, scope)
      case _: Accumulate[?] => Vector.empty

  private def warpWarnings(mask: Expr[?], span: SourceSpan, location: String,
      controlMayDiverge: Boolean, scope: UniformityScope): Vector[ValidationWarning] =
    val maskMayVary = isLaneVarying(mask, scope)
    if controlMayDiverge || maskMayVary then
      val reasons = Vector(
        Option.when(controlMayDiverge)("lane-varying control flow"),
        Option.when(maskMayVary)("a lane-varying mask")
      ).flatten.mkString(" and ")
      Vector(ValidationWarning(ValidationWarningCode.WarpParticipationMayDiverge,
        s"warp collective uses $reasons; verify that every calling lane belongs to the mask " +
          "and every non-exited named lane reaches the same call with the same mask", location, span))
    else Vector.empty

  private def isLaneVarying(expression: Expr[?], scope: UniformityScope): Boolean =
    UniformityAnalysis.expression(expression, scope) == Uniformity.Varying

  private def isKnownDivergent(
      expression: Expr[?],
      scope: UniformityScope
  ): Boolean =
    UniformityAnalysis.expression(expression, scope) match
      case Uniformity.WarpUniform | Uniformity.Varying => true
      case Uniformity.GridUniform | Uniformity.BlockUniform | Uniformity.Unknown =>
        false
