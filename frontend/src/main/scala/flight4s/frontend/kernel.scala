package flight4s.frontend

import scala.annotation.{experimental, MacroAnnotation}
import scala.quoted.*
import flight4s.core.dsl.{CudaDsl, DslSourcePosition}
import flight4s.core.ir.{Expr as DeviceExpr, Kernel, Load, Local, LocalVariable, SourceSpan}

/** Opt-in prototype for typed kernel factories. */
@experimental
final class kernel extends MacroAnnotation:
  override def transform(using Quotes)(definition: quotes.reflect.Definition,
      companion: Option[quotes.reflect.Definition]): List[quotes.reflect.Definition] =
    import quotes.reflect.*
    definition match
      case method: DefDef if method.rhs.nonEmpty =>
        if method.paramss.exists {
            case TermParamClause(parameters) => parameters.nonEmpty
            case _ => true
          } || method.symbol.flags.is(Flags.Inline) then
          report.errorAndAbort("@kernel requires a non-inline factory with no parameters", method.pos)
        if !(method.returnTpt.tpe <:< TypeRepr.of[Kernel[?]]) then
          report.errorAndAbort("@kernel factory must return Kernel[Args]", method.returnTpt.pos)
        val rhs = method.rhs.get
        val kernelMethods = TypeRepr.of[CudaDsl.type].typeSymbol.methodMember("kernel")
        val letMethods = TypeRepr.of[CudaDsl.type].typeSymbol.methodMember("let")
        def transformFactory(term: Term): Term = term match
          case Inlined(call, Nil, expansion) =>
            Inlined.copy(term)(call, Nil, transformFactory(expansion))
          case Typed(expression, tpt) => Typed.copy(term)(transformFactory(expression), tpt)
          case call @ Apply(function, List(body)) if kernelMethods.contains(function.symbol) =>
            Apply.copy(call)(function, List(transformBody(body)))
          case _ => report.errorAndAbort("@kernel factory must directly call CudaDsl.kernel with an inline body", term.pos)

        def transformBody(term: Term): Term = term match
          case Inlined(call, Nil, expansion) => Inlined.copy(term)(call, Nil, transformBody(expansion))
          case block @ Block(Nil, result) => Block.copy(block)(Nil, transformBody(result))
          case block @ Block(List(function: DefDef), closure: Closure) if function.rhs.nonEmpty =>
            val parameters = function.termParamss.flatMap(_.params)
            val builders = parameters.filter(_.tpt.tpe =:= TypeRepr.of[CudaDsl.BlockBuilder])
            val body = if builders.size == 1 then
              transformStatements(function.rhs.get, builders.head.symbol, Map.empty)
            else transformBody(function.rhs.get)
            val transformed = DefDef.copy(function)(function.name, function.paramss, function.returnTpt, Some(body))
            Block.copy(block)(List(transformed), closure)
          case _ => report.errorAndAbort("@kernel requires a literal kernel-body lambda", term.pos)

        def explicitLet(term: Term): Boolean = letMethods.contains(calledSymbol(term))

        def elementType(tpe: TypeRepr): Option[TypeRepr] =
          val base = tpe.baseType(TypeRepr.of[DeviceExpr[Any]].typeSymbol)
          base match
            case AppliedType(_, List(element)) => Some(element)
            case _ => None

        def deviceType(value: ValDef): Option[TypeRepr] = elementType(value.tpt.tpe)

        def hostLiteral(term: Term): Boolean = term match
          case Literal(_) => true
          case _ => false

        def calledSymbol(term: Term): Symbol = term match
          case Apply(function, _) => calledSymbol(function)
          case TypeApply(function, _) => calledSymbol(function)
          case Inlined(_, _, expansion) => calledSymbol(expansion)
          case _ => term.symbol

        def libraryOperation(symbol: Symbol): Boolean =
          val name = symbol.fullName
          name.startsWith("flight4s.core.dsl.") || name.startsWith("flight4s.core.ir.") ||
            name.startsWith("flight4s.core.types.") || name.startsWith("scala.Tuple") ||
            name.startsWith("scala.runtime.Tuples.") || symbol.name == "$asInstanceOf$"

        def localSymbol(symbol: Symbol): Boolean =
          var owner = symbol.owner
          while owner != Symbol.noSymbol && owner != method.symbol do owner = owner.owner
          owner == method.symbol

        def statementBuilder(function: DefDef): Option[Symbol] =
          function.termParamss.flatMap(_.params) match
            case List(parameter) if function.rhs.nonEmpty &&
                parameter.tpt.tpe =:= TypeRepr.of[CudaDsl.BlockBuilder] &&
                function.returnTpt.tpe =:= TypeRepr.of[Unit] => Some(parameter.symbol)
            case _ => None

        def checkNested(tree: Tree, variables: Map[Symbol, Symbol]): Unit =
          val check = new TreeTraverser:
            override def traverseTree(tree: Tree)(owner: Symbol): Unit =
              tree match
                // Each statement callback is checked separately with its own builder.
                case Block(List(function: DefDef), closure: Closure) if closure.meth.symbol == function.symbol =>
                  if statementBuilder(function).isEmpty then
                    function.rhs.foreach(body => traverseTree(body)(function.symbol))
                case value: ValDef if value.symbol.flags.is(Flags.Mutable) || value.symbol.flags.is(Flags.Lazy) =>
                  report.errorAndAbort("@kernel mutable locals require a statement-producing DSL body; lazy vals are not supported", value.pos)
                case value: ValDef if value.rhs.nonEmpty && deviceType(value).nonEmpty && !value.rhs.exists(explicitLet) =>
                  report.errorAndAbort("@kernel implicit snapshots require a statement-producing DSL body; expression-only callbacks remain pure", value.pos)
                case _: Assign =>
                  report.errorAndAbort("@kernel mutable assignments require a statement-producing DSL body; expression-only callbacks remain pure", tree.pos)
                case _: If | _: Match | _: While | _: Try | _: Return =>
                  report.errorAndAbort("@kernel prototype requires explicit DSL control flow and assignment", tree.pos)
                case call: Apply if !libraryOperation(calledSymbol(call)) =>
                  report.errorAndAbort("@kernel prototype supports DSL operations only; host/helper calls are not translated", call.pos)
                case reference: Ref if reference.symbol.flags.is(Flags.Mutable) && !variables.contains(reference.symbol) =>
                  report.errorAndAbort("@kernel prototype cannot capture mutable host state", reference.pos)
                case reference: Ref if elementType(reference.tpe).nonEmpty &&
                    !localSymbol(reference.symbol) && !libraryOperation(reference.symbol) =>
                  report.errorAndAbort("@kernel prototype cannot capture external Expr values; declare them inside the kernel body", reference.pos)
                case reference: Ref if reference.symbol.flags.is(Flags.Method) && !libraryOperation(reference.symbol) =>
                  report.errorAndAbort("@kernel prototype supports DSL operations only; host/helper calls are not translated", reference.pos)
                case _ => traverseTreeChildren(tree)(owner)
          check.traverseTree(tree)(method.symbol)

        def nestedRewriter(tree: Tree, variables: Map[Symbol, Symbol]): TreeMap =
          checkNested(tree, variables)
          new TreeMap:
            override def transformTerm(term: Term)(owner: Symbol): Term = term match
              case block @ Block(List(function: DefDef), closure: Closure) if
                  closure.meth.symbol == function.symbol && statementBuilder(function).nonEmpty =>
                val body = transformStatements(function.rhs.get, statementBuilder(function).get, variables)
                val transformed = DefDef.copy(function)(function.name, function.paramss, function.returnTpt, Some(body))
                Block.copy(block)(List(transformed), closure)
              case reference: Ref if variables.contains(reference.symbol) =>
                elementType(reference.tpe).get.asType match
                  case '[t] =>
                    given Quotes = owner.asQuotes
                    val place = Ref(variables(reference.symbol)).asExprOf[LocalVariable[t]]
                    val span = sourceSpan(reference.pos)
                    '{ Load($place, $span): DeviceExpr[t] }.asTerm
              case selection: Select =>
                val qualifier = transformTerm(selection.qualifier)(owner)
                // Select.copy round-trips structured compiler names through String.
                if qualifier == selection.qualifier then selection
                else Select(qualifier, selection.symbol)
              case _ => super.transformTerm(term)(owner)

        def transformNested(tree: Term, variables: Map[Symbol, Symbol], owner: Symbol): Term =
          nestedRewriter(tree, variables).transformTerm(tree)(owner)

        def transformNestedStatement(tree: Statement, variables: Map[Symbol, Symbol], owner: Symbol): Statement =
          nestedRewriter(tree, variables).transformStatement(tree)(owner)

        def sourceSpan(position: Position): Expr[SourceSpan] =
          '{ SourceSpan(${Expr(position.sourceFile.path)}, ${Expr(position.startLine + 1)},
            ${Expr(position.startColumn + 1)}, ${Expr(position.endLine + 1)}, ${Expr(position.endColumn + 1)}) }

        def snapshot(value: ValDef, builder: Symbol, variables: Map[Symbol, Symbol]): ValDef =
          val rhs = transformNested(value.rhs.get, variables, value.symbol)
          deviceType(value).get.asType match
            case '[t] =>
              if !(value.tpt.tpe =:= TypeRepr.of[DeviceExpr[t]]) then
                report.errorAndAbort("@kernel val must have Expr[T] type, not a concrete IR node subtype", value.pos)
              given Quotes = value.symbol.asQuotes
              val initial = rhs.asExprOf[DeviceExpr[t]]
              val block = Ref(builder).asExprOf[CudaDsl.BlockBuilder]
              val span = sourceSpan(value.pos)
              val transformed = '{
                val initialValue = $initial
                CudaDsl.let(initialValue)(using initialValue.valueType, $block, DslSourcePosition($span))
              }
              ValDef.copy(value)(value.name, value.tpt, Some(transformed.asTerm))

        def mutableLocal(value: ValDef, builder: Symbol, variables: Map[Symbol, Symbol]): ValDef =
          deviceType(value) match
            case Some(element) if value.rhs.nonEmpty => element.asType match
              case '[t] =>
                if !(value.tpt.tpe =:= TypeRepr.of[DeviceExpr[t]]) then
                  report.errorAndAbort("@kernel var must have Expr[T] type, not a concrete IR node subtype", value.pos)
                val storage = Symbol.newVal(value.symbol.owner, Symbol.freshName(s"${value.name}_device"),
                  TypeRepr.of[LocalVariable[t]], Flags.EmptyFlags, Symbol.noSymbol)
                val rhs = transformNested(value.rhs.get, variables, value.symbol).changeOwner(storage)
                val transformed =
                  given Quotes = storage.asQuotes
                  val initial = rhs.asExprOf[DeviceExpr[t]]
                  val block = Ref(builder).asExprOf[CudaDsl.BlockBuilder]
                  val span = sourceSpan(value.pos)
                  '{
                    val initialValue = $initial
                    CudaDsl.local(initialValue)(using initialValue.valueType, $block, DslSourcePosition($span))
                  }
                ValDef(storage, Some(transformed.asTerm))
            case _ =>
              report.errorAndAbort("@kernel mutable locals require initialized Expr[T] bindings; host vars are not translated", value.pos)

        def deviceAssignment(assignment: Assign, builder: Symbol, variables: Map[Symbol, Symbol]): Term =
          assignment.lhs match
            case reference: Ref if variables.contains(reference.symbol) =>
              elementType(reference.tpe).get.asType match
                case '[t] =>
                  val rhs = transformNested(assignment.rhs, variables, builder.owner)
                  given Quotes = builder.owner.asQuotes
                  val place = Ref(variables(reference.symbol)).asExprOf[LocalVariable[t]]
                  val initial = rhs.asExprOf[DeviceExpr[t]]
                  val block = Ref(builder).asExprOf[CudaDsl.BlockBuilder]
                  val span = sourceSpan(assignment.pos)
                  '{ CudaDsl.:=[t, Local]($place)($initial)(using $block, DslSourcePosition($span)) }.asTerm
            case _ =>
              report.errorAndAbort("@kernel assignment must target a mutable device local declared in a statement body; host mutation is not translated", assignment.pos)

        def transformStatements(term: Term, builder: Symbol, inherited: Map[Symbol, Symbol]): Term = term match
          case block @ Block(statements, result) =>
            // Only earlier lexical declarations are visible; sibling callbacks cannot leak state.
            var variables = inherited
            val transformed = statements.map {
              case value: ValDef if value.symbol.flags.is(Flags.Lazy) =>
                report.errorAndAbort("@kernel prototype does not support lazy val", value.pos)
              case value: ValDef if value.symbol.flags.is(Flags.Mutable) =>
                val declaration = mutableLocal(value, builder, variables)
                variables = variables.updated(value.symbol, declaration.symbol)
                declaration
              case value: ValDef if deviceType(value).nonEmpty && !value.rhs.exists(explicitLet) =>
                snapshot(value, builder, variables)
              case value: ValDef if !value.tpt.tpe.typeSymbol.fullName.startsWith("flight4s.core.") &&
                  !value.rhs.exists(hostLiteral) =>
                report.errorAndAbort("@kernel prototype does not support this binding type; use explicit DSL bindings", value.pos)
              case assignment: Assign => deviceAssignment(assignment, builder, variables)
              case statement =>
                transformNestedStatement(statement, variables, builder.owner)
            }
            val finalResult = result match
              case assignment: Assign => deviceAssignment(assignment, builder, variables)
              case other => transformNested(other, variables, builder.owner)
            Block.copy(block)(transformed, finalResult)
          case assignment: Assign => deviceAssignment(assignment, builder, inherited)
          case other =>
            transformNested(other, inherited, builder.owner)

        List(DefDef.copy(method)(method.name, method.paramss, method.returnTpt, Some(transformFactory(rhs))))
      case _ =>
        report.errorAndAbort("@kernel supports only concrete kernel factory methods", definition.pos)
