package flight4s.ml

/** Strict reader and writer for exactly the versioned MLP model schema; not a general JSON library.
  * Doubles are written with java.lang.Double.toString, whose shortest form round-trips exactly. */
private[ml] object MlpJson:
  def write(model: MlpModel): String =
    MlpModel.validate(model)
    val builder = StringBuilder()
    builder ++= "{\n"
    builder ++= s"""  "format": "${MlpModel.FormatName}",\n"""
    builder ++= s"""  "version": ${MlpModel.FormatVersion},\n"""
    builder ++= s"""  "inputSize": ${model.inputSize},\n"""
    builder ++= s"""  "hiddenSize": ${model.hiddenSize},\n"""
    builder ++= s"""  "activation": "${MlpModel.Activation}",\n"""
    builder ++= s"""  "output": "${MlpModel.Output}",\n"""
    builder ++= s"""  "precision": "${MlpModel.Precision}",\n"""
    builder ++= s"""  "w1": ${numbers(model.w1)},\n"""
    builder ++= s"""  "b1": ${numbers(model.b1)},\n"""
    builder ++= s"""  "w2": ${numbers(model.w2)},\n"""
    builder ++= s"""  "b2": ${model.b2}\n"""
    builder ++= "}\n"
    builder.result()

  private def numbers(values: Array[Double]): String =
    values.mkString("[", ", ", "]")

  def read(text: String): MlpModel =
    val fields = Parser(text).parseRoot()
    def field(name: String): Value =
      fields.getOrElse(name, throw IllegalArgumentException(s"model JSON is missing the field '$name'"))
    def stringField(name: String): String = field(name) match
      case Value.Str(value) => value
      case _ => throw IllegalArgumentException(s"model JSON field '$name' must be a string")
    def intField(name: String): Int = field(name) match
      case Value.Num(value) if value.isWhole && value >= Int.MinValue && value <= Int.MaxValue => value.toInt
      case _ => throw IllegalArgumentException(s"model JSON field '$name' must be a whole number")
    def numberField(name: String): Double = field(name) match
      case Value.Num(value) => value
      case _ => throw IllegalArgumentException(s"model JSON field '$name' must be a number")
    def arrayField(name: String): Array[Double] = field(name) match
      case Value.Arr(values) => values.toArray
      case _ => throw IllegalArgumentException(s"model JSON field '$name' must be an array of numbers")
    MlValidation.requireThat(stringField("format") == MlpModel.FormatName,
      s"unsupported model format '${stringField("format")}'; expected '${MlpModel.FormatName}'")
    MlValidation.requireThat(intField("version") == MlpModel.FormatVersion,
      s"unsupported model format version ${intField("version")}; expected ${MlpModel.FormatVersion}")
    MlValidation.requireThat(stringField("activation") == MlpModel.Activation,
      s"unsupported activation '${stringField("activation")}'")
    MlValidation.requireThat(stringField("output") == MlpModel.Output,
      s"unsupported output convention '${stringField("output")}'")
    MlValidation.requireThat(stringField("precision") == MlpModel.Precision,
      s"unsupported precision '${stringField("precision")}'")
    val model = MlpModel(intField("inputSize"), intField("hiddenSize"),
      arrayField("w1"), arrayField("b1"), arrayField("w2"), numberField("b2"))
    MlpModel.validate(model)
    model

  private enum Value:
    case Str(value: String)
    case Num(value: Double)
    case Arr(values: Vector[Double])

  /** Recursive-descent reader for the schema's subset: one flat object of strings,
    * numbers and number arrays, without escapes; anything else is a clear error. */
  private final class Parser(text: String):
    private var position = 0

    private def fail(message: String): Nothing =
      throw IllegalArgumentException(s"invalid model JSON at offset $position: $message")

    private def skipWhitespace(): Unit =
      while position < text.length && text.charAt(position).isWhitespace do position += 1

    private def peek(): Char =
      skipWhitespace()
      if position >= text.length then fail("unexpected end of input")
      text.charAt(position)

    private def expect(expected: Char): Unit =
      if peek() != expected then fail(s"expected '$expected' but found '${text.charAt(position)}'")
      position += 1

    private def parseString(): String =
      expect('"')
      val start = position
      while position < text.length && text.charAt(position) != '"' do
        if text.charAt(position) == '\\' then fail("string escapes are not part of this schema")
        position += 1
      if position >= text.length then fail("unterminated string")
      val result = text.substring(start, position)
      position += 1
      result

    private def parseNumber(): Double =
      skipWhitespace()
      val start = position
      while position < text.length && "+-0123456789.eE".indexOf(text.charAt(position).toInt) >= 0 do position += 1
      if position == start then fail("expected a number")
      val token = text.substring(start, position)
      try java.lang.Double.parseDouble(token)
      catch case _: NumberFormatException => fail(s"malformed number '$token'")

    private def parseNumberArray(): Vector[Double] =
      expect('[')
      val builder = Vector.newBuilder[Double]
      if peek() == ']' then position += 1
      else
        var open = true
        while open do
          builder += parseNumber()
          peek() match
            case ',' => position += 1
            case ']' =>
              position += 1
              open = false
            case other => fail(s"expected ',' or ']' but found '$other'")
      builder.result()

    private def parseValue(): Value = peek() match
      case '"' => Value.Str(parseString())
      case '[' => Value.Arr(parseNumberArray())
      case _ => Value.Num(parseNumber())

    def parseRoot(): Map[String, Value] =
      expect('{')
      val fields = Map.newBuilder[String, Value]
      var open = peek() != '}'
      if !open then position += 1
      while open do
        val key = parseString()
        expect(':')
        fields += key -> parseValue()
        peek() match
          case ',' => position += 1
          case '}' =>
            position += 1
            open = false
          case other => fail(s"expected ',' or '}' but found '$other'")
      skipWhitespace()
      if position != text.length then fail("unexpected content after the closing brace")
      fields.result()
