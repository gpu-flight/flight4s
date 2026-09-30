package flight4s.frontend

import scala.annotation.experimental
import scala.quoted.*
import flight4s.core.dsl.{CudaDsl, DslSourcePosition}
import flight4s.core.ir.{Expr as DeviceExpr, *}
import flight4s.core.types.*

@experimental
private[frontend] object ScalaKernelMacro:
  def build[Args <: Tuple: Type, Params <: Tuple: Type](name: Expr[String],
      signature: Expr[KernelSignature[Args] { type Bindings = Params }],
      body: Expr[ScalaKernel.DeviceBindingsOf[Params] => Unit])(using Quotes)
      : Expr[Kernel[Args]] =
    import quotes.reflect.*

    def unwrapped(term: Term): Term = term match
      case Inlined(_, Nil, expansion) => unwrapped(expansion)
      case Typed(value, _) => unwrapped(value)
      case Block(Nil, value) => unwrapped(value)
      case other => other

    val (parameter, sourceBody) = unwrapped(body.asTerm) match
      case Lambda(List(parameter), sourceBody) => (parameter.symbol, sourceBody)
      case other => report.errorAndAbort("ScalaKernel requires a literal kernel-body lambda", other.pos)

    def primitive(tpe: TypeRepr): Boolean =
      Vector(TypeRepr.of[Int], TypeRepr.of[Float], TypeRepr.of[Double], TypeRepr.of[Boolean])
        .exists(_ =:= tpe.widen.dealias)

    def tupleElements(tpe: TypeRepr): List[TypeRepr] = tpe.dealias.asType match
      case '[EmptyTuple] => Nil
      case '[head *: tail] => TypeRepr.of[head] :: tupleElements(TypeRepr.of[tail])
      case _ => report.errorAndAbort("ScalaKernel requires a concrete parameter tuple", body)

    val parameterTypes = tupleElements(TypeRepr.of[Params])
    parameterTypes.foreach { tpe => tpe.asType match
      case '[ScalarParam[t]] if primitive(TypeRepr.of[t]) => ()
      case '[BufferParam[t, mode]] if primitive(TypeRepr.of[t]) &&
          (TypeRepr.of[mode] =:= TypeRepr.of[ReadOnly] || TypeRepr.of[mode] =:= TypeRepr.of[ReadWrite]) => ()
      case _ => report.errorAndAbort("ScalaKernel supports Int, Float, Double and Boolean parameters only", body)
    }

    def valueType[T: Type](using Quotes): Expr[CudaType[T]] =
      val result = Type.of[T] match
        case '[Int] => '{ I32 }
        case '[Float] => '{ F32 }
        case '[Double] => '{ F64 }
        case '[Boolean] => '{ Bool }
        case _ => report.errorAndAbort("ScalaKernel supports Int, Float, Double and Boolean values only", body)
      result.asExprOf[CudaType[T]]

    def span(position: Position)(using Quotes): Expr[SourceSpan] =
      '{ SourceSpan(${Expr(position.sourceFile.path)}, ${Expr(position.startLine + 1)},
        ${Expr(position.startColumn + 1)}, ${Expr(position.endLine + 1)}, ${Expr(position.endColumn + 1)}) }

    def position(term: Tree)(using Quotes): Expr[DslSourcePosition] = '{ DslSourcePosition(${span(term.pos)}) }

    final case class Binding(handle: Term, mutable: Boolean = false)
    type Environment = Map[Symbol, Binding]

    def invocation(term: Term): Option[(Select, List[List[Term]])] = unwrapped(term) match
      case Apply(function, arguments) => invocation(function).map { (selection, lists) => (selection, lists :+ arguments) }
      case TypeApply(function, _) => invocation(function)
      case selection: Select => Some((selection, Nil))
      case _ => None

    val arraySymbol = TypeRepr.of[ScalaKernel.DeviceArray[Any, ReadOnly]].typeSymbol
    val readMethods = arraySymbol.methodMember("apply")
    val writeMethods = arraySymbol.methodMember("update")
    val intrinsicSymbols = List(
      TypeRepr.of[ScalaKernel.threadIdx.type], TypeRepr.of[ScalaKernel.blockIdx.type],
      TypeRepr.of[ScalaKernel.blockDim.type], TypeRepr.of[ScalaKernel.gridDim.type])
      .flatMap(tpe => List("x", "y", "z").flatMap(axis =>
        tpe.typeSymbol.methodMember(axis).map(_ -> s"${tpe.typeSymbol.name.stripSuffix("$")}.$axis"))).toMap

    def parameterReference(term: Term): Boolean = unwrapped(term) match
      case reference: Ref => reference.symbol == parameter
      // The typer narrows match-type tuples before selecting their elements.
      case TypeApply(Select(reference: Ref, "$asInstanceOf$"), List(tpt)) =>
        reference.symbol == parameter && tpt.tpe =:= TypeRepr.of[ScalaKernel.DeviceBindingsOf[Params]]
      case _ => false

    def binding(term: Term, env: Environment, bindings: Expr[Params])(using Quotes): Option[Binding] = unwrapped(term) match
      case reference: Ident => env.get(reference.symbol)
      case selection @ Select(qualifier, name) if parameterReference(qualifier) &&
          name.matches("_[1-9][0-9]*") && selection.symbol.owner.fullName.startsWith("scala.Tuple") =>
        val index = name.drop(1).toInt - 1
        if index >= parameterTypes.size then report.errorAndAbort("invalid kernel parameter index", selection.pos)
        parameterTypes(index).asType match
          case '[p] => Some(Binding('{ $bindings.productElement(${Expr(index)}).asInstanceOf[p] }.asTerm))
      // Tupled lambdas use the standard library's inlined Tuple.apply(index).
      case Inlined(Some(call: Term), List(alias: ValDef), _) if alias.rhs.exists(parameterReference) => invocation(call) match
        case Some((selection, List(List(Literal(IntConstant(index)))))) if
            selection.symbol.owner.fullName == "scala.Tuple" && selection.name == "apply" &&
            index >= 0 && index < parameterTypes.size =>
          parameterTypes(index).asType match
            case '[p] => Some(Binding('{ $bindings.productElement(${Expr(index)}).asInstanceOf[p] }.asTerm))
        case _ => None
      case _ => None

    def expression[T: Type](term: Term, env: Environment, bindings: Expr[Params])(using Quotes): Expr[DeviceExpr[T]] =
      val source = unwrapped(term)
      val sourceType = source.tpe.widen.dealias
      if !(sourceType <:< TypeRepr.of[T]) || !primitive(TypeRepr.of[T]) then
        report.errorAndAbort("ScalaKernel requires matching primitive operand types; numeric conversions are not supported yet", source.pos)
      val location = span(source.pos)
      val cudaType = valueType[T]
      binding(source, env, bindings) match
        case Some(bound) => bound.handle.tpe.widen.asType match
          case '[LocalVariable[t]] =>
            val handle = bound.handle.asExprOf[LocalVariable[T]]
            '{ Load($handle, $location) }
          case '[ScalarParam[t]] => bound.handle.asExprOf[DeviceExpr[T]]
          case _ => report.errorAndAbort("a device buffer is not a scalar value", source.pos)
        case None => source match
          case _: Block => report.errorAndAbort(
            "ScalaKernel expression blocks are not supported yet; branches must remain pure", source.pos)
          case Literal(constant) =>
            val literal = Literal(constant).asExprOf[T]
            '{ flight4s.core.ir.Literal($literal, $cudaType, $location) }
          case conditional: If =>
            val condition = expression[Boolean](conditional.cond, env, bindings)
            val yes = expression[T](conditional.thenp, env, bindings)
            val no = expression[T](conditional.elsep, env, bindings)
            '{ Conditional($condition, $yes, $no, $cudaType, $location) }
          case _ => invocation(source) match
            case Some((selection, Nil)) if intrinsicSymbols.contains(selection.symbol) =>
              '{ Intrinsic(${Expr(intrinsicSymbols(selection.symbol))}, I32, $location) }.asExprOf[DeviceExpr[T]]
            case Some((selection, List(List(index)))) if readMethods.contains(selection.symbol) =>
              val array = binding(selection.qualifier, env, bindings).getOrElse(
                report.errorAndAbort("device array access must reference a kernel parameter", selection.pos))
              val offset = expression[Int](index, env, bindings)
              array.handle.tpe.widen.asType match
                case '[BufferParam[t, mode]] =>
                  val handle = array.handle.asExprOf[BufferParam[T, mode]]
                  '{ Load(BufferElement[T, mode]($handle.name, $offset, $cudaType, $location), $location) }
                case _ => report.errorAndAbort("device array access must reference a buffer parameter", selection.pos)
            case Some((selection, arguments)) if
                Set("scala.Int", "scala.Float", "scala.Double", "scala.Boolean", "scala.Any").contains(selection.symbol.owner.fullName) =>
              primitiveOperation[T](selection, arguments.flatten, source, env, bindings)
            case _ => report.errorAndAbort(
              s"ScalaKernel does not translate this expression, host/helper calls or captures: ${source.show}", source.pos)

    def primitiveOperation[T: Type](selection: Select, arguments: List[Term], source: Term,
        env: Environment, bindings: Expr[Params])(using Quotes): Expr[DeviceExpr[T]] =
      val location = span(source.pos)
      val operandType = selection.qualifier.tpe.widen.dealias
      val numeric = !(operandType =:= TypeRepr.of[Boolean])
      val arithmetic = Map("+" -> BinaryOperator.Add, "-" -> BinaryOperator.Subtract,
        "*" -> BinaryOperator.Multiply, "/" -> BinaryOperator.Divide, "%" -> BinaryOperator.Remainder,
        "&" -> BinaryOperator.BitAnd, "|" -> BinaryOperator.BitOr, "^" -> BinaryOperator.BitXor)
      val comparisons = Map("<" -> ComparisonOperator.LessThan, "<=" -> ComparisonOperator.LessThanOrEqual,
        ">" -> ComparisonOperator.GreaterThan, ">=" -> ComparisonOperator.GreaterThanOrEqual,
        "==" -> ComparisonOperator.Equal, "!=" -> ComparisonOperator.NotEqual)
      val operation = selection.name
      if arithmetic.contains(operation) && numeric && arguments.size == 1 &&
          (!Set("%", "&", "|", "^").contains(operation) || operandType =:= TypeRepr.of[Int]) then
        val left = expression[T](selection.qualifier, env, bindings)
        val right = expression[T](arguments.head, env, bindings)
        val operator = arithmetic(operation) match
          case BinaryOperator.Add => '{ BinaryOperator.Add }
          case BinaryOperator.Subtract => '{ BinaryOperator.Subtract }
          case BinaryOperator.Multiply => '{ BinaryOperator.Multiply }
          case BinaryOperator.Divide => '{ BinaryOperator.Divide }
          case BinaryOperator.Remainder => '{ BinaryOperator.Remainder }
          case BinaryOperator.BitAnd => '{ BinaryOperator.BitAnd }
          case BinaryOperator.BitOr => '{ BinaryOperator.BitOr }
          case BinaryOperator.BitXor => '{ BinaryOperator.BitXor }
        '{ Binary($operator, $left, $right, ${valueType[T]}, $location) }
      else if comparisons.contains(operation) && arguments.size == 1 &&
          (numeric || Set("==", "!=").contains(operation)) then
        operandType.asType match
          case '[t] =>
            val left = expression[t](selection.qualifier, env, bindings)
            val right = expression[t](arguments.head, env, bindings)
            val operator = comparisons(operation) match
              case ComparisonOperator.LessThan => '{ ComparisonOperator.LessThan }
              case ComparisonOperator.LessThanOrEqual => '{ ComparisonOperator.LessThanOrEqual }
              case ComparisonOperator.GreaterThan => '{ ComparisonOperator.GreaterThan }
              case ComparisonOperator.GreaterThanOrEqual => '{ ComparisonOperator.GreaterThanOrEqual }
              case ComparisonOperator.Equal => '{ ComparisonOperator.Equal }
              case ComparisonOperator.NotEqual => '{ ComparisonOperator.NotEqual }
            '{ Compare($operator, $left, $right, ${valueType[t]}, $location) }.asExprOf[DeviceExpr[T]]
      else if Set("&&", "||", "unary_!").contains(operation) && operandType =:= TypeRepr.of[Boolean] &&
          arguments.size == (if operation == "unary_!" then 0 else 1) then
        val left = expression[Boolean](selection.qualifier, env, bindings)
        val no = '{ flight4s.core.ir.Literal(false, Bool, $location): DeviceExpr[Boolean] }
        val yes = '{ flight4s.core.ir.Literal(true, Bool, $location): DeviceExpr[Boolean] }
        val result = operation match
          case "&&" => '{ Conditional($left, ${expression[Boolean](arguments.head, env, bindings)}, $no, Bool, $location) }
          case "||" => '{ Conditional($left, $yes, ${expression[Boolean](arguments.head, env, bindings)}, Bool, $location) }
          case _ => '{ Conditional($left, $no, $yes, Bool, $location) }
        result.asExprOf[DeviceExpr[T]]
      else if Set("unary_-", "unary_+").contains(operation) && numeric && arguments.isEmpty &&
          (operation == "unary_+" || operandType =:= TypeRepr.of[Int]) then
        val operand = expression[T](selection.qualifier, env, bindings)
        if operation == "unary_+" then operand
        else
          val zero = '{ flight4s.core.ir.Literal(0, I32, $location) }
          '{ Binary(BinaryOperator.Subtract, ${zero.asExprOf[DeviceExpr[T]]}, $operand, ${valueType[T]}, $location) }
      else report.errorAndAbort("ScalaKernel does not support this primitive operation or conversion yet", source.pos)

    def statement(term: Term, env: Environment, bindings: Expr[Params], builder: Expr[CudaDsl.BlockBuilder])(using Quotes): Expr[Unit] =
      val source = unwrapped(term)
      val location = span(source.pos)
      source match
        case Literal(UnitConstant()) => '{ () }
        case branch: If =>
          val condition = expression[Boolean](branch.cond, env, bindings)
          val yes: Expr[CudaDsl.BlockBuilder ?=> Unit] = '{ (nested: CudaDsl.BlockBuilder) ?=>
            ${statements(branch.thenp, env, bindings, 'nested)} }
          val noAlternative = unwrapped(branch.elsep) match
            case Literal(UnitConstant()) => true
            case _ => false
          if noAlternative then
            '{ CudaDsl.when($condition)($yes)(using $builder, ${position(source)}) }
          else
            val no: Expr[CudaDsl.BlockBuilder ?=> Unit] = '{ (nested: CudaDsl.BlockBuilder) ?=>
              ${statements(branch.elsep, env, bindings, 'nested)} }
            '{ CudaDsl.gpuIf($condition)($yes)($no)(using $builder, ${position(source)}) }
        case block: Block =>
          val nestedBody: Expr[CudaDsl.BlockBuilder ?=> Unit] = '{ (nested: CudaDsl.BlockBuilder) ?=>
            ${statements(block, env, bindings, 'nested)} }
          '{ CudaDsl.scoped($nestedBody)(using $builder, ${position(source)}) }
        case assignment: Assign => assignment.lhs match
          case reference: Ref if env.get(reference.symbol).exists(_.mutable) =>
            assignment.rhs.tpe.widen.asType match
              case '[t] =>
                val handle = env(reference.symbol).handle.asExprOf[LocalVariable[t]]
                val value = expression[t](assignment.rhs, env, bindings)
                '{ CudaDsl.:=[t, Local]($handle)($value)(using $builder, ${position(source)}) }
          case _ => report.errorAndAbort("ScalaKernel assignments must target a device local; host mutation is not supported", source.pos)
        case _ => invocation(source) match
          case Some((selection, List(List(index, value), _))) if writeMethods.contains(selection.symbol) =>
            val array = binding(selection.qualifier, env, bindings).getOrElse(
              report.errorAndAbort("device array writes must reference a kernel parameter", selection.pos))
            array.handle.tpe.widen.asType match
              case '[BufferParam[t, ReadWrite]] =>
                val handle = array.handle.asExprOf[BufferParam[t, ReadWrite]]
                val offset = expression[Int](index, env, bindings)
                val initial = expression[t](value, env, bindings)
                '{ CudaDsl.:=[t, Global](BufferElement[t, ReadWrite]($handle.name, $offset,
                    $handle.valueType, $location))($initial)(using $builder, ${position(source)}) }
              case _ => report.errorAndAbort("ScalaKernel writes require an output buffer", source.pos)
          case _ => report.errorAndAbort("ScalaKernel supports assignments and if statements only; loops and host effects are not supported yet", source.pos)

    def statements(term: Term, env: Environment, bindings: Expr[Params], builder: Expr[CudaDsl.BlockBuilder])(using Quotes): Expr[Unit] =
      def next(pending: List[Statement], result: Term, current: Environment)(using Quotes): Expr[Unit] = pending match
        case Nil => statement(result, current, bindings, builder)
        case (value: ValDef) :: tail =>
          if value.symbol.flags.is(Flags.Lazy) || value.rhs.isEmpty then
            report.errorAndAbort("ScalaKernel locals must be initialized and cannot be lazy", value.pos)
          val rhs = value.rhs.get
          val mutable = value.symbol.flags.is(Flags.Mutable)
          val isBuffer = value.tpt.tpe.widen.baseType(arraySymbol) match
            case AppliedType(_, _) => true
            case _ => false
          if isBuffer && !mutable then
            val original = binding(rhs, current, bindings).getOrElse(
              report.errorAndAbort(s"buffer aliases must reference kernel parameters: ${rhs.show}", rhs.pos))
            next(tail, result, current.updated(value.symbol, original))
          else
            val declaredType = value.tpt.tpe.widen.dealias
            if !primitive(declaredType) then report.errorAndAbort("ScalaKernel supports primitive locals and immutable buffer aliases only", value.pos)
            declaredType.asType match
              case '[t] =>
                val initial = expression[t](rhs, current, bindings)
                val valuePosition = position(value)
                '{
                  val handle = CudaDsl.local($initial)(using ${valueType[t]}, $builder, $valuePosition)
                  ${next(tail, result, current.updated(value.symbol, Binding('handle.asTerm, mutable)))}
                }
        case (term: Term) :: tail =>
          '{ ${statement(term, current, bindings, builder)}; ${next(tail, result, current)} }
        case unsupported :: _ => report.errorAndAbort("ScalaKernel does not support local definitions or imports inside the body", unsupported.pos)
      unwrapped(term) match
        case Block(pending, result) => next(pending, result, env)
        case other => statement(other, env, bindings, builder)

    '{
      val kernelName = $name
      val kernelSignature = $signature
      CudaDsl.kernel(kernelName, kernelSignature) { bindings =>
        (builder: CudaDsl.BlockBuilder) ?=> ${statements(sourceBody, Map.empty, 'bindings, 'builder)}
      }
    }
