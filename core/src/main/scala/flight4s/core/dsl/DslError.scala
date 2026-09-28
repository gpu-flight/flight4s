package flight4s.core.dsl

import flight4s.core.ir.SourceSpan

enum DslErrorCode:
  case SharedMemoryDeclarationOutsideKernelBody
  case StatementInsideExpression
  case InvalidWarpReductionGroup

final case class DslError(
    code: DslErrorCode,
    message: String,
    span: SourceSpan = SourceSpan.Unknown
) extends RuntimeException(message)
