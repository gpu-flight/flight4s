package flight4s.core.ir

private[ir] object CudaIdentifier:
  private val syntax = raw"[A-Za-z_][A-Za-z0-9_]*".r
  private val vectorTypeNames = Set("float2", "float4")

  def isValid(value: String): Boolean =
    syntax.matches(value) && !vectorTypeNames.contains(value)
