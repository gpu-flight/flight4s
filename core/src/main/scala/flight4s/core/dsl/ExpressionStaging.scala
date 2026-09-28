package flight4s.core.dsl

import flight4s.core.ir.SourceSpan

private[dsl] object ExpressionStaging:
  // A callback can capture any outer builder, so an instance-local flag is insufficient.
  private val active = ThreadLocal.withInitial[Boolean](() => false)

  def expression[T](body: => T): T =
    if active.get() then body
    else
      active.set(true)
      try body
      finally active.remove()

  def requireStatementsAllowed(span: SourceSpan): Unit =
    if active.get() then
      throw DslError(
        DslErrorCode.StatementInsideExpression,
        "expression builders cannot emit DSL statements or shared declarations; " +
          "stage them in the enclosing block or a when/gpuIf/foreach body",
        span
      )
