package flight4s.core.ir

private[ir] object CudaIdentifier:
  private val syntax = raw"[A-Za-z_][A-Za-z0-9_]*".r
  private val vectorTypeNames = Set("float2", "float4")
  private val keywords = Set(
    "alignas", "alignof", "asm", "auto", "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "concept", "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await", "co_return",
    "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export",
    "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new",
    "noexcept", "nullptr", "operator", "private", "protected", "public", "register", "reinterpret_cast", "requires",
    "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "template",
    "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "wchar_t", "while", "and", "and_eq", "bitand", "bitor", "compl", "not", "not_eq",
    "or", "or_eq", "xor", "xor_eq"
  )

  def isValid(value: String): Boolean =
    syntax.matches(value) && !vectorTypeNames.contains(value) && !keywords.contains(value)
