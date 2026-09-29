package flight4s.frontend

import scala.annotation.{experimental, MacroAnnotation}
import scala.quoted.*
import flight4s.core.dsl.{CudaDsl, DslSourcePosition}
import flight4s.core.ir.{Expr as DeviceExpr, Kernel, SourceSpan}

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
              transformStatements(function.rhs.get, builders.head.symbol)
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

        def checkNested(tree: Tree): Unit =
          val check = new TreeTraverser:
            override def traverseTree(tree: Tree)(owner: Symbol): Unit =
              tree match
                case value: ValDef if value.symbol.flags.is(Flags.Mutable) || value.symbol.flags.is(Flags.Lazy) =>
                  report.errorAndAbort("@kernel prototype does not support Scala var or lazy val; use explicit DSL state", value.pos)
                case value: ValDef if deviceType(value).nonEmpty && !value.rhs.exists(explicitLet) =>
                  report.errorAndAbort("@kernel prototype snapshots only top-level Expr vals; use explicit let in nested bodies", value.pos)
                case _: If | _: Match | _: While | _: Try | _: Return | _: Assign =>
                  report.errorAndAbort("@kernel prototype requires explicit DSL control flow and assignment", tree.pos)
                case call: Apply if !libraryOperation(calledSymbol(call)) =>
                  report.errorAndAbort("@kernel prototype supports DSL operations only; host/helper calls are not translated", call.pos)
                case reference: Ref if reference.symbol.flags.is(Flags.Mutable) =>
                  report.errorAndAbort("@kernel prototype cannot capture mutable host state", reference.pos)
                case reference: Ref if elementType(reference.tpe).nonEmpty &&
                    !localSymbol(reference.symbol) && !libraryOperation(reference.symbol) =>
                  report.errorAndAbort("@kernel prototype cannot capture external Expr values; declare them inside the kernel body", reference.pos)
                case reference: Ref if reference.symbol.flags.is(Flags.Method) && !libraryOperation(reference.symbol) =>
                  report.errorAndAbort("@kernel prototype supports DSL operations only; host/helper calls are not translated", reference.pos)
                case _ => traverseTreeChildren(tree)(owner)
          check.traverseTree(tree)(method.symbol)

        def snapshot(value: ValDef, builder: Symbol): ValDef =
          val rhs = value.rhs.get
          checkNested(rhs)
          deviceType(value).get.asType match
            case '[t] =>
              if !(value.tpt.tpe =:= TypeRepr.of[DeviceExpr[t]]) then
                report.errorAndAbort("@kernel val must have Expr[T] type, not a concrete IR node subtype", value.pos)
              given Quotes = value.symbol.asQuotes
              val initial = rhs.asExprOf[DeviceExpr[t]]
              val block = Ref(builder).asExprOf[CudaDsl.BlockBuilder]
              val position = value.pos
              val span = '{ SourceSpan(${Expr(position.sourceFile.path)}, ${Expr(position.startLine + 1)},
                ${Expr(position.startColumn + 1)}, ${Expr(position.endLine + 1)}, ${Expr(position.endColumn + 1)}) }
              val transformed = '{
                val initialValue = $initial
                CudaDsl.let(initialValue)(using initialValue.valueType, $block, DslSourcePosition($span))
              }
              ValDef.copy(value)(value.name, value.tpt, Some(transformed.asTerm))

        def transformStatements(term: Term, builder: Symbol): Term = term match
          case block @ Block(statements, result) =>
            val transformed = statements.map {
              case value: ValDef if value.symbol.flags.is(Flags.Mutable) || value.symbol.flags.is(Flags.Lazy) =>
                report.errorAndAbort("@kernel prototype does not support Scala var or lazy val; use explicit DSL state", value.pos)
              case value: ValDef if deviceType(value).nonEmpty && !value.rhs.exists(explicitLet) =>
                snapshot(value, builder)
              case value: ValDef if !value.tpt.tpe.typeSymbol.fullName.startsWith("flight4s.core.") &&
                  !value.rhs.exists(hostLiteral) =>
                report.errorAndAbort("@kernel prototype does not support this binding type; use explicit DSL bindings", value.pos)
              case statement =>
                checkNested(statement)
                statement
            }
            checkNested(result)
            Block.copy(block)(transformed, result)
          case other =>
            checkNested(other)
            other

        List(DefDef.copy(method)(method.name, method.paramss, method.returnTpt, Some(transformFactory(rhs))))
      case _ =>
        report.errorAndAbort("@kernel supports only concrete kernel factory methods", definition.pos)
