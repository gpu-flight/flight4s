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

    val namedTupleSymbol = TypeRepr.of[(field: Int)].typeSymbol
    def tupleValueType(tpe: TypeRepr): TypeRepr = tpe.widen.dealias match
      case AppliedType(constructor, List(_, values)) if constructor.typeSymbol == namedTupleSymbol => values
      case other => other

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

    sealed trait ScopedBinding
    final case class Binding(handle: Term, mutable: Boolean = false) extends ScopedBinding
    final case class TupleBinding(fields: List[Binding]) extends ScopedBinding
    final case class TraversalBinding(plan: TraversalPlan) extends ScopedBinding
    type Environment = Map[Symbol, ScopedBinding]
    enum TraversalOperation:
      case Mapping, Guard, Flattening
    final case class TraversalStage(input: Symbol, body: Term, environment: Environment,
        resultType: TypeRepr, operation: TraversalOperation)
    final case class TraversalPlan(from: Expr[DeviceExpr[Int]], until: Expr[DeviceExpr[Int]],
        stages: List[TraversalStage], elementType: TypeRepr)

    def invocation(term: Term): Option[(Select, List[List[Term]])] = unwrapped(term) match
      case Apply(function, arguments) => invocation(function).map { (selection, lists) => (selection, lists :+ arguments) }
      case TypeApply(function, _) => invocation(function)
      case selection: Select => Some((selection, Nil))
      case _ => None

    val arraySymbol = TypeRepr.of[ScalaKernel.DeviceArray[Any, ReadOnly]].typeSymbol
    val readMethods = arraySymbol.methodMember("apply")
    val writeMethods = arraySymbol.methodMember("update")
    val sharedSymbol = TypeRepr.of[ScalaKernel.DeviceSharedArray[Any]].typeSymbol
    val sharedReadMethods = sharedSymbol.methodMember("apply")
    val sharedWriteMethods = sharedSymbol.methodMember("update")
    val sharedArrayMethods = TypeRepr.of[ScalaKernel.type].typeSymbol.methodMember("sharedArray")
    val barrierMethods = TypeRepr.of[ScalaKernel.type].typeSymbol.methodMember("barrier")
    val markerModule = Symbol.requiredModule("flight4s.frontend.ScalaKernel")

    // Imported markers are Idents; qualified calls must not erase an effectful receiver.
    def markerArguments(term: Term, methods: List[Symbol]): Option[List[List[Term]]] = unwrapped(term) match
      case Apply(function, arguments) => markerArguments(function, methods).map(_ :+ arguments)
      case TypeApply(function, _) => markerArguments(function, methods)
      case reference: Ident if methods.contains(reference.symbol) => Some(Nil)
      case selection: Select if methods.contains(selection.symbol) && selection.qualifier.symbol == markerModule => Some(Nil)
      case _ => None

    val rangeForeachMethods = TypeRepr.of[scala.collection.immutable.Range].typeSymbol.methodMember("foreach")
    val filteredRangeType = TypeRepr.of[scala.collection.WithFilter[Int, Iterable]]
    val filteredForeachMethods = filteredRangeType.typeSymbol.methodMember("foreach")
    val withFilterMethods = TypeRepr.of[scala.collection.immutable.Range].typeSymbol.methodMember("withFilter") ++
      filteredRangeType.typeSymbol.methodMember("withFilter")
    val untilMethods = TypeRepr.of[scala.runtime.RichInt].typeSymbol.methodMember("until")
    val intWrapperMethods = Symbol.requiredModule("scala.Predef").methodMember("intWrapper")
    val deviceRangeMethods = TypeRepr.of[ScalaKernel.type].typeSymbol.methodMember("deviceRange")
    val traversalSymbol = TypeRepr.of[ScalaKernel.DeviceTraversal[Int]].typeSymbol
    val traversalMapMethods = traversalSymbol.methodMember("map")
    val traversalFlatMapMethods = traversalSymbol.methodMember("flatMap")
    val traversalGuardMethods = traversalSymbol.methodMember("withFilter")
    val traversalForeachMethods = traversalSymbol.methodMember("foreach")
    val traversalFoldMethods = traversalSymbol.methodMember("foldLeft")
    val tupleConstructors = (1 to 22).flatMap(arity =>
      Symbol.requiredModule(s"scala.Tuple$arity").methodMember("apply")).toSet
    val namedTupleModule = Symbol.requiredModule("scala.NamedTuple")
    val namedTupleBuildMethods = namedTupleModule.methodMember("build")
    val namedTupleApplyMethods = namedTupleModule.methodMember("apply")
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
      case reference: Ident => env.get(reference.symbol).collect { case bound: Binding => bound }
      // Named fields expand to the standard NamedTuple.apply with a literal index.
      case Inlined(Some(call: Term), List(_: ValDef, receiver: ValDef), _) if invocation(call).exists { (selection, _) =>
          namedTupleApplyMethods.contains(selection.symbol) && selection.qualifier.symbol == namedTupleModule
        } => invocation(call) match
        case Some((_, List(List(_), List(Literal(IntConstant(index)))))) => receiver.rhs.map(unwrapped) match
          // Inline call traces keep old symbols; the proxy RHS retains the current lexical owner.
          case Some(TypeApply(Select(reference: Ident, "$asInstanceOf$"), List(_))) =>
            env.get(reference.symbol).collect { case tuple: TupleBinding => tuple }.flatMap(_.fields.lift(index))
          case _ => None
        case _ => None
      case selection @ Select(reference: Ident, name) if name.matches("_[1-9][0-9]*") &&
          selection.symbol.owner.fullName.startsWith("scala.Tuple") &&
          env.get(reference.symbol).exists(_.isInstanceOf[TupleBinding]) => env.get(reference.symbol) match
        case Some(TupleBinding(fields)) => fields.lift(name.drop(1).toInt - 1)
        case _ => None
      case selection @ Select(qualifier, name) if parameterReference(qualifier) &&
          name.matches("_[1-9][0-9]*") && selection.symbol.owner.fullName.startsWith("scala.Tuple") =>
        val index = name.drop(1).toInt - 1
        if index >= parameterTypes.size then report.errorAndAbort("invalid kernel parameter index", selection.pos)
        parameterTypes(index).asType match
          case '[p] => Some(Binding('{ $bindings.productElement(${Expr(index)}).asInstanceOf[p] }.asTerm))
      // Tupled lambdas use the standard library's inlined Tuple.apply(index).
      case Inlined(Some(call: Term), List(alias: ValDef), _) => invocation(call) match
        case Some((selection, List(List(Literal(IntConstant(index)))))) if
            selection.symbol.owner.fullName == "scala.Tuple" && selection.name == "apply" =>
          alias.rhs.map(unwrapped) match
            case Some(reference: Ident) if env.get(reference.symbol).exists(_.isInstanceOf[TupleBinding]) =>
              env(reference.symbol).asInstanceOf[TupleBinding].fields.lift(index)
            case Some(rhs) if parameterReference(rhs) && index >= 0 && index < parameterTypes.size =>
              parameterTypes(index).asType match
                case '[p] => Some(Binding('{ $bindings.productElement(${Expr(index)}).asInstanceOf[p] }.asTerm))
            case _ => None
        case _ => None
      case _ => None

    def readBinding[T: Type](bound: Binding, location: Expr[SourceSpan])(using Quotes): Expr[DeviceExpr[T]] =
      bound.handle.tpe.widen.asType match
        case '[LocalVariable[t]] =>
          val handle = bound.handle.asExprOf[LocalVariable[T]]
          '{ Load($handle, $location) }
        case '[ScalarParam[t]] => bound.handle.asExprOf[DeviceExpr[T]]
        case '[DeviceExpr[t]] => bound.handle.asExprOf[DeviceExpr[T]]
        case _ => report.errorAndAbort("a device buffer is not a scalar value", body)

    def elementTypes(tpe: TypeRepr, source: Term): List[TypeRepr] =
      if primitive(tpe) then List(tpe)
      else if tupleValueType(tpe) <:< TypeRepr.of[Tuple] then
        val fields = tupleElements(tupleValueType(tpe))
        tpe.widen.dealias match
          case AppliedType(constructor, List(names, _)) if constructor.typeSymbol == namedTupleSymbol =>
            val nameTypes = tupleElements(names)
            val labels = nameTypes.collect { case ConstantType(StringConstant(name)) => name }
            if labels.size != nameTypes.size || labels.size != fields.size || labels.distinct.size != labels.size then
              report.errorAndAbort("ScalaKernel named tuple labels must be unique literal strings matching the field count", source.pos)
          case _ => ()
        if fields.nonEmpty && fields.size <= 22 && fields.forall(primitive) then fields
        else report.errorAndAbort("ScalaKernel traversal callbacks require pure primitive results or flat nonempty primitive tuples of at most 22 fields", source.pos)
      else report.errorAndAbort("ScalaKernel traversal callbacks require pure primitive results or flat nonempty primitive tuples of at most 22 fields", source.pos)

    def placeholderElement(tpe: TypeRepr, source: Term)(using Quotes): ScopedBinding =
      val fields = elementTypes(tpe, source).map { field => field.asType match
        case '[t] => Binding('{ Intrinsic("traversal_callback", ${valueType[t]}, ${span(source.pos)}): DeviceExpr[t] }.asTerm)
      }
      if primitive(tpe) then fields.head else TupleBinding(fields)

    def callbackBody(term: Term, env: Environment, bindings: Expr[Params])(using Quotes): (Term, Environment) =
      unwrapped(term) match
        case Block(pending, result) if pending.nonEmpty =>
          // Auto-tupling introduces immutable field projections before a pure callback body.
          val projections = pending.foldLeft(Option(env)) { (current, statement) =>
            for
              scope <- current
              value <- statement match
                case value: ValDef if !value.symbol.flags.is(Flags.Mutable) && !value.symbol.flags.is(Flags.Lazy) => Some(value)
                case _ => None
              rhs <- value.rhs
              bound <- binding(rhs, scope, bindings)
              if primitive(value.tpt.tpe.widen.dealias)
              if scope.values.exists {
                case TupleBinding(fields) => fields.contains(bound)
                case _ => false
              }
            yield scope.updated(value.symbol, bound)
          }
          projections.map(result -> _).getOrElse(term -> env)
        case _ => term -> env

    def elementValues(term: Term, tpe: TypeRepr, env: Environment, bindings: Expr[Params])(
        using Quotes): List[(TypeRepr, Term)] =
      val source = unwrapped(term)
      val types = elementTypes(tpe, source)
      val (translated, current) = callbackBody(source, env, bindings)
      if primitive(tpe) then tpe.asType match
        case '[t] => List(tpe -> expression[t](translated, current, bindings).asTerm)
      else
        val existing = unwrapped(translated) match
          case reference: Ident => current.get(reference.symbol).collect { case tuple: TupleBinding => tuple }
          case _ => None
        existing match
          case Some(tuple) => types.zip(tuple.fields).map { (field, bound) => field.asType match
            case '[t] => field -> readBinding[t](bound, span(source.pos)).asTerm
          }
          case None =>
            // Inspect only the known library build call, never arbitrary inline expansion bodies.
            val constructor = unwrapped(translated) match
              case Inlined(Some(call: Term), List(_: ValDef, _: ValDef, values: ValDef), _) => invocation(call) match
                case Some((selection, List(Nil, List(_)))) if
                    namedTupleBuildMethods.contains(selection.symbol) && selection.qualifier.symbol == namedTupleModule =>
                  values.rhs.getOrElse(translated)
                case _ => translated
              case _ => translated
            invocation(constructor) match
              case Some((selection, List(arguments))) if tupleConstructors.contains(selection.symbol) && arguments.size == types.size =>
                types.zip(arguments).map { (field, argument) => field.asType match
                  case '[t] => field -> expression[t](argument, current, bindings).asTerm
                }
              case _ => report.errorAndAbort("ScalaKernel tuple values require a direct standard tuple constructor or an existing tuple binding", source.pos)

    def snapshotElement(term: Term, tpe: TypeRepr, env: Environment, bindings: Expr[Params],
        builder: Expr[CudaDsl.BlockBuilder])(consume: Quotes ?=> ScopedBinding => Expr[Unit])(
        using Quotes): Expr[Unit] =
      def next(pending: List[(TypeRepr, Term)], fields: List[Binding])(using Quotes): Expr[Unit] = pending match
        case Nil => consume(if primitive(tpe) then fields.head else TupleBinding(fields))
        case (field, value) :: tail => field.asType match
          case '[t] =>
            val initial = value.asExprOf[DeviceExpr[t]]
            '{
              val handle = CudaDsl.local($initial)(using ${valueType[t]}, $builder, ${position(term)})
              ${next(tail, fields :+ Binding('handle.asTerm))}
            }
      // Snapshot fields left to right before guards or terminal stores can observe them.
      next(elementValues(term, tpe, env, bindings), Nil)

    def expression[T: Type](term: Term, env: Environment, bindings: Expr[Params])(using Quotes): Expr[DeviceExpr[T]] =
      val source = unwrapped(term)
      val sourceType = source.tpe.widen.dealias
      if !(sourceType <:< TypeRepr.of[T]) || !primitive(TypeRepr.of[T]) then
        report.errorAndAbort("ScalaKernel requires matching primitive operand types; numeric conversions are not supported yet", source.pos)
      val location = span(source.pos)
      val cudaType = valueType[T]
      binding(source, env, bindings) match
        case Some(bound) => readBinding[T](bound, location)
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
            case Some((selection, _)) if traversalFoldMethods.contains(selection.symbol) =>
              report.errorAndAbort("ScalaKernel foldLeft must directly initialize a primitive val or var, or an immutable tuple val; embedded folds are not supported yet", source.pos)
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
            case Some((selection, List(List(index)))) if sharedReadMethods.contains(selection.symbol) =>
              val array = binding(selection.qualifier, env, bindings).getOrElse(
                report.errorAndAbort("shared array reads must reference a declared shared array or immutable alias", selection.pos))
              array.handle.tpe.widen.asType match
                case '[SharedArray[t, Rank1]] =>
                  val handle = array.handle.asExprOf[SharedArray[T, Rank1]]
                  val offset = expression[Int](index, env, bindings)
                  '{ Load(SharedElement($handle.name, Vector($offset), $cudaType, $location), $location) }
                case _ => report.errorAndAbort("shared array reads require shared storage", selection.pos)
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

    def traversal(term: Term, env: Environment, bindings: Expr[Params], builder: Expr[CudaDsl.BlockBuilder])(
        consume: Quotes ?=> TraversalPlan => Expr[Unit])(using Quotes): Expr[Unit] =
      val source = unwrapped(term)
      source match
        case reference: Ident => env.get(reference.symbol) match
          case Some(TraversalBinding(plan)) => consume(plan)
          case _ => report.errorAndAbort("ScalaKernel traversal plans cannot capture host values", source.pos)
        case Apply(function, List(from, until)) if deviceRangeMethods.contains(function.symbol) =>
          val initial = expression[Int](from, env, bindings)
          val limit = expression[Int](until, env, bindings)
          // Plan construction captures bounds; terminal reuse does not recapture them.
          '{
            val start = CudaDsl.local($initial)(using I32, $builder, ${position(from)})
            val end = CudaDsl.local($limit)(using I32, $builder, ${position(until)})
            ${consume(TraversalPlan('{ Load(start, ${span(from.pos)}) },
              '{ Load(end, ${span(until.pos)}) }, Nil, TypeRepr.of[Int]))}
          }
        case _ => invocation(source) match
          case Some((selection, List(List(callback)))) if
              traversalMapMethods.contains(selection.symbol) || traversalGuardMethods.contains(selection.symbol) ||
                traversalFlatMapMethods.contains(selection.symbol) =>
            traversal(selection.qualifier, env, bindings, builder) { plan =>
              val (input, body) = unwrapped(callback) match
                case Lambda(List(input), body) if input.tpt.tpe.widen.dealias =:= plan.elementType => (input.symbol, body)
                case _ => report.errorAndAbort("ScalaKernel traversals require a literal matching primitive callback lambda", callback.pos)
              val guard = traversalGuardMethods.contains(selection.symbol)
              val flatten = traversalFlatMapMethods.contains(selection.symbol)
              // The method's inferred result already widens literal unions to the declared scalar type.
              val resultType = if guard then TypeRepr.of[Boolean] else source.tpe.widen.baseType(traversalSymbol) match
                case AppliedType(_, List(element)) => element.widen.dealias
                case _ => report.errorAndAbort("ScalaKernel requires a concrete traversal element type", source.pos)
              elementTypes(resultType, body)
              // Validate even an unused plan, without staging or executing its device mapping.
              val current = env.updated(input, placeholderElement(plan.elementType, callback))
              if flatten then
                val (factory, factoryEnv) = callbackBody(body, current, bindings)
                traversal(factory, factoryEnv, bindings, builder) { inner =>
                  if !(inner.elementType =:= resultType) then
                    report.errorAndAbort("ScalaKernel flatMap requires a matching traversal result", body.pos)
                  '{ () }
                }
              else elementValues(body, resultType, current, bindings)
              val operation = if guard then TraversalOperation.Guard
                else if flatten then TraversalOperation.Flattening else TraversalOperation.Mapping
              consume(plan.copy(stages = plan.stages :+ TraversalStage(input, body, env, resultType, operation),
                elementType = if guard then plan.elementType else resultType))
            }
          case _ => report.errorAndAbort("ScalaKernel traversal plans must originate from deviceRange; host collections and captures are not supported", source.pos)

    def traversalElement(stages: List[TraversalStage], element: ScopedBinding, bindings: Expr[Params],
        builder: Expr[CudaDsl.BlockBuilder])(consume: Quotes ?=> (ScopedBinding, Expr[CudaDsl.BlockBuilder]) => Expr[Unit])(
        using Quotes): Expr[Unit] = stages match
      case Nil => consume(element, builder)
      case stage :: tail =>
        val current = stage.environment.updated(stage.input, element)
        if stage.operation == TraversalOperation.Guard then
          val (predicate, predicateEnv) = callbackBody(stage.body, current, bindings)
          val condition = expression[Boolean](predicate, predicateEnv, bindings)
          val guarded: Expr[CudaDsl.BlockBuilder ?=> Unit] = '{ (nested: CudaDsl.BlockBuilder) ?=>
            ${traversalElement(tail, element, bindings, 'nested)(consume)} }
          '{ CudaDsl.when($condition)($guarded)(using $builder, ${position(stage.body)}) }
        else if stage.operation == TraversalOperation.Flattening then
          // Construct the inner plan after outer guards, once per accepted outer item.
          val (factory, factoryEnv) = callbackBody(stage.body, current, bindings)
          traversal(factory, factoryEnv, bindings, builder) { inner =>
            traverse(inner, bindings, builder, stage.body) { (item, active) =>
              traversalElement(tail, item, bindings, active)(consume)
            }
          }
        else snapshotElement(stage.body, stage.resultType, current, bindings, builder) { mapped =>
          traversalElement(tail, mapped, bindings, builder)(consume)
        }

    def traverse(plan: TraversalPlan, bindings: Expr[Params], builder: Expr[CudaDsl.BlockBuilder], source: Term)(
        consume: Quotes ?=> (ScopedBinding, Expr[CudaDsl.BlockBuilder]) => Expr[Unit])(using Quotes): Expr[Unit] =
      '{
        CudaDsl.gpuFor(${plan.from}, ${plan.until}) {
          (index: DeviceExpr[Int]) => (nested: CudaDsl.BlockBuilder) ?=>
            ${traversalElement(plan.stages, Binding('index.asTerm), bindings, 'nested)(consume)}
        }(using $builder, ${position(source)})
      }

    def initializer[T: Type](term: Term, env: Environment, bindings: Expr[Params], builder: Expr[CudaDsl.BlockBuilder])(
        consume: Quotes ?=> Expr[DeviceExpr[T]] => Expr[Unit])(using Quotes): Expr[Unit] =
      val source = unwrapped(term)
      invocation(source) match
        case Some((selection, List(List(initial), List(callback)))) if traversalFoldMethods.contains(selection.symbol) =>
          if !(source.tpe.widen.dealias =:= TypeRepr.of[T]) then
            report.errorAndAbort("ScalaKernel foldLeft requires a matching primitive accumulator type", source.pos)
          traversal(selection.qualifier, env, bindings, builder) { plan =>
            val (state, input, step) = unwrapped(callback) match
              case Lambda(List(state, input), step) if state.tpt.tpe.widen.dealias =:= TypeRepr.of[T] &&
                  input.tpt.tpe.widen.dealias =:= plan.elementType => (state.symbol, input.symbol, step)
              case _ => report.errorAndAbort("ScalaKernel foldLeft requires a literal matching two-parameter primitive step lambda", callback.pos)
            val seed = expression[T](initial, env, bindings)
            // Bounds are captured before the seed; each terminal gets its own ordered state.
            '{
              val accumulator = CudaDsl.local($seed)(using ${valueType[T]}, $builder, ${position(initial)})
              ${traverse(plan, bindings, builder, source) { (element, active) =>
                val current = env.updated(state, Binding('accumulator.asTerm)).updated(input, element)
                val (stepBody, stepEnv) = callbackBody(step, current, bindings)
                val updated = expression[T](stepBody, stepEnv, bindings)
                '{ CudaDsl.:=[T, Local](accumulator)($updated)(using $active, ${position(step)}) }
              }}
              ${consume('{ Load(accumulator, ${span(source.pos)}) })}
            }
          }
        case _ => consume(expression[T](source, env, bindings))

    def tupleInitializer(term: Term, stateType: TypeRepr, env: Environment,
        bindings: Expr[Params], builder: Expr[CudaDsl.BlockBuilder])(
        consume: Quotes ?=> TupleBinding => Expr[Unit])(using Quotes): Expr[Unit] =
      val source = unwrapped(term)
      val fields = elementTypes(stateType, source)
      invocation(source) match
        case Some((selection, List(List(initial), List(callback)))) if traversalFoldMethods.contains(selection.symbol) =>
          if !(source.tpe.widen.dealias =:= stateType) then
            report.errorAndAbort("ScalaKernel foldLeft requires a matching tuple accumulator type", source.pos)
          traversal(selection.qualifier, env, bindings, builder) { plan =>
            val (state, input, step) = unwrapped(callback) match
              case Lambda(List(state, input), step) if state.tpt.tpe.widen.dealias =:= stateType &&
                  input.tpt.tpe.widen.dealias =:= plan.elementType => (state.symbol, input.symbol, step)
              case _ => report.errorAndAbort("ScalaKernel foldLeft requires a literal matching two-parameter step lambda", callback.pos)
            snapshotElement(initial, stateType, env, bindings, builder) {
              case accumulator: TupleBinding =>
                '{
                  ${traverse(plan, bindings, builder, source) { (element, active) =>
                    val current = env.updated(state, accumulator).updated(input, element)
                    snapshotElement(step, stateType, current, bindings, active) {
                      case updated: TupleBinding =>
                        // Every next field must read the previous state before any field is assigned.
                        fields.zip(accumulator.fields.zip(updated.fields)).foldRight('{ () }) {
                          case ((field, (target, value)), remaining) => field.asType match
                            case '[t] =>
                              val local = target.handle.asExprOf[LocalVariable[t]]
                              val next = readBinding[t](value, span(step.pos))
                              '{
                                CudaDsl.:=[t, Local]($local)($next)(using $active, ${position(step)})
                                $remaining
                              }
                        }
                      case _ => report.errorAndAbort("ScalaKernel foldLeft requires a tuple next state", step.pos)
                    }
                  }}
                  ${consume(accumulator)}
                }
              case _ => report.errorAndAbort("ScalaKernel foldLeft requires a tuple seed", initial.pos)
            }
          }
        case _ => report.errorAndAbort("ScalaKernel tuple locals must directly initialize an immutable foldLeft result", source.pos)

    def statement(term: Term, env: Environment, bindings: Expr[Params], builder: Expr[CudaDsl.BlockBuilder],
        sharedRoot: Option[Expr[CudaDsl.BlockBuilder]] = None)(using Quotes): Expr[Unit] =
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
          // Unconditional scopes retain allocation ownership; branches and loops do not.
          val nestedBody: Expr[CudaDsl.BlockBuilder ?=> Unit] = '{ (nested: CudaDsl.BlockBuilder) ?=>
            ${statements(block, env, bindings, 'nested, sharedRoot)} }
          '{ CudaDsl.scoped($nestedBody)(using $builder, ${position(source)}) }
        case assignment: Assign => assignment.lhs match
          case reference: Ref if binding(reference, env, bindings).exists(_.mutable) =>
            assignment.rhs.tpe.widen.asType match
              case '[t] =>
                val handle = binding(reference, env, bindings).get.handle.asExprOf[LocalVariable[t]]
                val value = expression[t](assignment.rhs, env, bindings)
                '{ CudaDsl.:=[t, Local]($handle)($value)(using $builder, ${position(source)}) }
          case _ => report.errorAndAbort("ScalaKernel assignments must target a device local; host mutation is not supported", source.pos)
        case _ if markerArguments(source, barrierMethods).contains(List(Nil)) =>
          '{ CudaDsl.barrier()(using $builder, ${position(source)}) }
        case _ => invocation(source) match
          case Some((selection, List(List(callback)))) if traversalForeachMethods.contains(selection.symbol) =>
            traversal(selection.qualifier, env, bindings, builder) { plan =>
              val (input, loopBody) = unwrapped(callback) match
                case Lambda(List(input), loopBody) if input.tpt.tpe.widen.dealias =:= plan.elementType => (input.symbol, loopBody)
                case _ => report.errorAndAbort("ScalaKernel traversals require a literal matching primitive callback lambda", callback.pos)
              traverse(plan, bindings, builder, source) { (element, active) =>
                statements(loopBody, env.updated(input, element), bindings, active)
              }
            }
          case Some((selection, List(List(callback)))) if
              rangeForeachMethods.contains(selection.symbol) || filteredForeachMethods.contains(selection.symbol) =>
            def range(term: Term): (Term, Term, List[(Symbol, Term)]) = invocation(term) match
              case Some((filter, List(List(predicate)))) if withFilterMethods.contains(filter.symbol) =>
                val (from, until, guards) = range(filter.qualifier)
                val guard = unwrapped(predicate) match
                  case Lambda(List(index), condition) if index.tpt.tpe =:= TypeRepr.of[Int] => (index.symbol, condition)
                  case _ => report.errorAndAbort("ScalaKernel range guards require a literal Int predicate lambda", predicate.pos)
                (from, until, guards :+ guard)
              case Some((bounds, List(List(end)))) if untilMethods.contains(bounds.symbol) =>
                unwrapped(bounds.qualifier) match
                  case Apply(wrapper, List(start)) if intWrapperMethods.contains(wrapper.symbol) => (start, end, Nil)
                  case _ => report.errorAndAbort("ScalaKernel range loops require Scala Int start until end", bounds.pos)
              case _ => report.errorAndAbort(
                "ScalaKernel range loops require direct start until end with unit stride; other range forms are not supported yet", term.pos)
            val (from, until, guards) = range(selection.qualifier)
            val (index, loopBody) = unwrapped(callback) match
              case Lambda(List(index), loopBody) if index.tpt.tpe =:= TypeRepr.of[Int] => (index.symbol, loopBody)
              case _ => report.errorAndAbort("ScalaKernel range loops require a literal Int loop-body lambda", callback.pos)
            val initial = expression[Int](from, env, bindings)
            val limit = expression[Int](until, env, bindings)
            def guarded(remaining: List[(Symbol, Term)], loopIndex: Expr[DeviceExpr[Int]],
                activeBuilder: Expr[CudaDsl.BlockBuilder])(using Quotes): Expr[Unit] = remaining match
              case Nil => statements(loopBody, env.updated(index, Binding(loopIndex.asTerm)), bindings, activeBuilder)
              case (guardIndex, predicate) :: tail =>
                val condition = expression[Boolean](predicate, env.updated(guardIndex, Binding(loopIndex.asTerm)), bindings)
                val body: Expr[CudaDsl.BlockBuilder ?=> Unit] = '{ (nested: CudaDsl.BlockBuilder) ?=>
                  ${guarded(tail, loopIndex, 'nested)} }
                // Chained withFilter predicates run in order, and later predicates must stay lazy.
                '{ CudaDsl.when($condition)($body)(using $activeBuilder, ${position(predicate)}) }
            // Scala constructs its Range before foreach; body stores must not change its bounds.
            '{
              val start = CudaDsl.local($initial)(using I32, $builder, ${position(from)})
              val end = CudaDsl.local($limit)(using I32, $builder, ${position(until)})
              CudaDsl.gpuFor(Load(start, ${span(from.pos)}), Load(end, ${span(until.pos)})) {
                (loopIndex: DeviceExpr[Int]) => (nested: CudaDsl.BlockBuilder) ?=>
                  ${guarded(guards, 'loopIndex, 'nested)}
              }(using $builder, ${position(source)})
            }
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
          case Some((selection, List(List(index, value)))) if sharedWriteMethods.contains(selection.symbol) =>
            val array = binding(selection.qualifier, env, bindings).getOrElse(
              report.errorAndAbort("shared array writes must reference a declared shared array or immutable alias", selection.pos))
            array.handle.tpe.widen.asType match
              case '[SharedArray[t, Rank1]] =>
                val handle = array.handle.asExprOf[SharedArray[t, Rank1]]
                val offset = expression[Int](index, env, bindings)
                val initial = expression[t](value, env, bindings)
                '{ CudaDsl.:=[t, Shared](SharedElement($handle.name, Vector($offset),
                    $handle.valueType, $location))($initial)(using $builder, ${position(source)}) }
              case _ => report.errorAndAbort("shared array writes require shared storage", selection.pos)
          case _ => report.errorAndAbort("ScalaKernel supports assignments, if statements and direct until range loops with guards only; other loops and host effects are not supported yet", source.pos)

    def statements(term: Term, env: Environment, bindings: Expr[Params], builder: Expr[CudaDsl.BlockBuilder],
        sharedRoot: Option[Expr[CudaDsl.BlockBuilder]] = None)(using Quotes): Expr[Unit] =
      def next(pending: List[Statement], result: Term, current: Environment)(using Quotes): Expr[Unit] = pending match
        case Nil => statement(result, current, bindings, builder, sharedRoot)
        case (value: ValDef) :: tail =>
          if value.symbol.flags.is(Flags.Lazy) || value.rhs.isEmpty then
            report.errorAndAbort("ScalaKernel locals must be initialized and cannot be lazy", value.pos)
          val rhs = value.rhs.get
          val mutable = value.symbol.flags.is(Flags.Mutable)
          val tupleAlias = unwrapped(rhs) match
            case reference: Ident => current.get(reference.symbol).collect { case tuple: TupleBinding => tuple }
            case _ => None
          val isBuffer = value.tpt.tpe.widen.baseType(arraySymbol) match
            case AppliedType(_, _) => true
            case _ => false
          val isTraversal = value.tpt.tpe.widen.baseType(traversalSymbol) match
            case AppliedType(_, _) => true
            case _ => false
          val declaredType = value.tpt.tpe.widen.dealias
          val isShared = declaredType.baseType(sharedSymbol) match
            case AppliedType(_, _) => true
            case _ => false
          val isTupleFold = tupleValueType(declaredType) <:< TypeRepr.of[Tuple] && invocation(rhs).exists {
            (selection, _) => traversalFoldMethods.contains(selection.symbol)
          }
          if isShared then
            if mutable then report.errorAndAbort("ScalaKernel shared array bindings must be immutable", value.pos)
            binding(rhs, current, bindings) match
              case Some(original) => next(tail, result, current.updated(value.symbol, original))
              case None => markerArguments(rhs, sharedArrayMethods) match
                case Some(List(List(size))) =>
                  val root = sharedRoot.getOrElse(report.errorAndAbort(
                    "ScalaKernel shared arrays must be declared outside branches and loops", value.pos))
                  val count = unwrapped(size) match
                    case Literal(IntConstant(count)) if count > 0 => count
                    case _ => report.errorAndAbort("ScalaKernel shared array size must be a positive compile-time Int constant", size.pos)
                  declaredType.asType match
                    case '[ScalaKernel.DeviceSharedArray[t]] if primitive(TypeRepr.of[t]) =>
                      '{
                        val handle = CudaDsl.sharedArray[t](${Expr(count)})(using ${valueType[t]}, $root, ${position(value)})
                        ${next(tail, result, current.updated(value.symbol, Binding('handle.asTerm)))}
                      }
                    case _ => report.errorAndAbort("ScalaKernel shared arrays support Int, Float, Double and Boolean elements only", value.pos)
                case _ => report.errorAndAbort("ScalaKernel shared arrays require a direct sharedArray declaration or immutable alias", rhs.pos)
          else if tupleAlias.isDefined && !mutable then
            next(tail, result, current.updated(value.symbol, tupleAlias.get))
          else if isTupleFold then
            if mutable then report.errorAndAbort("ScalaKernel tuple fold results must be immutable", value.pos)
            tupleInitializer(rhs, declaredType, current, bindings, builder) { initial =>
              next(tail, result, current.updated(value.symbol, initial))
            }
          else if isTraversal then
            if mutable then report.errorAndAbort("ScalaKernel traversal plans must be immutable", value.pos)
            traversal(rhs, current, bindings, builder) { plan =>
              next(tail, result, current.updated(value.symbol, TraversalBinding(plan)))
            }
          else if isBuffer && !mutable then
            val original = binding(rhs, current, bindings).getOrElse(
              report.errorAndAbort(s"buffer aliases must reference kernel parameters: ${rhs.show}", rhs.pos))
            next(tail, result, current.updated(value.symbol, original))
          else
            if !primitive(declaredType) then report.errorAndAbort("ScalaKernel supports primitive locals and immutable buffer aliases only", value.pos)
            declaredType.asType match
              case '[t] =>
                initializer[t](rhs, current, bindings, builder) { initial =>
                  '{
                    val handle = CudaDsl.local($initial)(using ${valueType[t]}, $builder, ${position(value)})
                    ${next(tail, result, current.updated(value.symbol, Binding('handle.asTerm, mutable)))}
                  }
                }
        case (term: Term) :: tail =>
          '{ ${statement(term, current, bindings, builder, sharedRoot)}; ${next(tail, result, current)} }
        case unsupported :: _ => report.errorAndAbort("ScalaKernel does not support local definitions or imports inside the body", unsupported.pos)
      unwrapped(term) match
        case Block(pending, result) => next(pending, result, env)
        case other => statement(other, env, bindings, builder, sharedRoot)

    '{
      val kernelName = $name
      val kernelSignature = $signature
      CudaDsl.kernel(kernelName, kernelSignature) { bindings =>
        (builder: CudaDsl.BlockBuilder) ?=> ${statements(sourceBody, Map.empty, 'bindings, 'builder, Some('builder))}
      }
    }
