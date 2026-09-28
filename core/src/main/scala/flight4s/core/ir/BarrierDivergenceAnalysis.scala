package flight4s.core.ir

private[core] object BarrierDivergenceAnalysis:
  def warnings(block: Block): Vector[ValidationWarning] =
    blockWarnings(
      block,
      "body",
      controlMayDiverge = false,
      UniformityScope.empty
    )

  private def blockWarnings(
      block: Block,
      location: String,
      controlMayDiverge: Boolean,
      initialScope: UniformityScope
  ): Vector[ValidationWarning] =
    block.statements.zipWithIndex
      .foldLeft((Vector.empty[ValidationWarning], initialScope)) {
        case ((warnings, scope), (statement, index)) =>
          val statementWarnings = warningsForStatement(
            statement,
            s"$location.statements[$index]",
            controlMayDiverge,
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
      scope: UniformityScope
  ): Vector[ValidationWarning] =
    statement match
      case branch: IfThen =>
        val branchMayDiverge =
          controlMayDiverge || isKnownDivergent(branch.condition, scope)
        blockWarnings(branch.thenBlock, s"$statementLocation.then", branchMayDiverge, scope) ++
          branch.elseBlock.toVector.flatMap(
            blockWarnings(_, s"$statementLocation.else", branchMayDiverge, scope)
          )

      case scoped: ScopedBlock =>
        blockWarnings(
          scoped.body,
          s"$statementLocation.body",
          controlMayDiverge,
          scope
        )

      case loop: ForLoop =>
        val loopMayDiverge =
          controlMayDiverge ||
            isKnownDivergent(loop.from, scope) ||
            isKnownDivergent(loop.until, scope)
        blockWarnings(
          loop.body,
          s"$statementLocation.body",
          loopMayDiverge,
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
      case _: LocalDeclaration[?] => Vector.empty
      case _: LocalArrayDeclaration[?] => Vector.empty
      case _: Store[?, ?] => Vector.empty
      case _: AtomicAdd[?, ?] => Vector.empty
      case _: AtomicFetchAdd[?, ?] => Vector.empty
      case _: Accumulate[?] => Vector.empty

  private def isKnownDivergent(
      expression: Expr[?],
      scope: UniformityScope
  ): Boolean =
    UniformityAnalysis.expression(expression, scope) match
      case Uniformity.WarpUniform | Uniformity.Varying => true
      case Uniformity.GridUniform | Uniformity.BlockUniform | Uniformity.Unknown =>
        false
