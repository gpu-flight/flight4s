package flight4s.core.frontend

import scala.annotation.experimental
import munit.FunSuite
import flight4s.frontend.examples.PrototypeKernels
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.{DslSourcePosition, CudaDsl}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

@experimental
class AnnotationIrSuite extends FunSuite:
  private def statements(block: Block): Vector[Stmt] = block.statements.flatMap {
    case branch: IfThen => Vector(branch) ++ statements(branch.thenBlock) ++ branch.elseBlock.toVector.flatMap(statements)
    case scope: ScopedBlock => Vector(scope) ++ statements(scope.body)
    case loop: ForLoop => Vector(loop) ++ statements(loop.body)
    case statement => Vector(statement)
  }

  test("mutable branches shadowing and loops exactly match explicit local load and store artifacts"):
    for branches <- Vector(true, false) do
      val actual = if branches then PrototypeKernels.mutableBranches else PrototypeKernels.mutableLoops
      val pending = scala.collection.mutable.Queue.from(statements(actual.body))
      def localLoads(expression: Expr[?]): Vector[Load[?, ?, ?]] = expression match
        case load: Load[?, ?, ?] => Vector(load)
        case binary: Binary[?] => localLoads(binary.left) ++ localLoads(binary.right)
        case _ => Vector.empty
      def read(variable: LocalVariable[Int]): Expr[Int] =
        val expression = pending.front match
          case declaration: LocalDeclaration[?] => declaration.initial
          case store: Store[?, ?] => store.value
          case other => fail(s"unexpected read at $other")
        val source = localLoads(expression).find(_.from == variable).get
        Load(variable, source.span)
      def declare(initial: => Expr[Int])(using builder: BlockBuilder): LocalVariable[Int] =
        val expression = initial
        val declaration = pending.dequeue().asInstanceOf[LocalDeclaration[Int]]
        local(declaration.local.name, expression)(using summon[flight4s.core.types.CudaType[Int]], builder,
          DslSourcePosition(declaration.span))
      def save(initial: => Expr[Int])(using builder: BlockBuilder): Expr[Int] =
        val expression = initial
        val declaration = pending.dequeue().asInstanceOf[LocalDeclaration[Int]]
        let(declaration.local.name, expression)(using summon[flight4s.core.types.CudaType[Int]], builder,
          DslSourcePosition(declaration.span))
      def write[Space <: AddressSpace](target: Place[Int, Space, ReadWrite], value: => Expr[Int])(using builder: BlockBuilder): Unit =
        val expression = value
        val store = pending.dequeue().asInstanceOf[Store[Int, Space]]
        target.:=(expression)(using builder, DslSourcePosition(store.span))
      val reference = CudaDsl.kernel(actual.name, actual.signature) { bindings =>
        val i = save(blockIdx.x * blockDim.x + threadIdx.x)
        val guard = pending.dequeue().asInstanceOf[IfThen]
        if branches then
          val p = bindings.asInstanceOf[(BufferParam[Int, ReadOnly], BufferParam[Int, ReadWrite],
            BufferParam[Int, ReadWrite], BufferParam[Int, ReadWrite], ScalarParam[Int])]
          when(i < p._5) {
            val total = declare(p._1(i).read)
            val before = save(read(total))
            val alternative = pending.dequeue().asInstanceOf[IfThen]
            gpuIf((i % literal(2)) === literal(0)) {
              write(total, read(total) + literal(3))
            } {
              write(total, read(total) - literal(2))
            }(using summon[BlockBuilder], DslSourcePosition(alternative.span))
            val scope = pending.dequeue().asInstanceOf[ScopedBlock]
            scoped {
              val inner = declare(literal(100) + i)
              write(inner, read(inner) + literal(1))
              write(p._4(i), read(inner))
            }(using summon[BlockBuilder], DslSourcePosition(scope.span))
            write(p._2(i), read(total))
            write(p._3(i), before)
          }(using summon[BlockBuilder], DslSourcePosition(guard.span))
        else
          val p = bindings.asInstanceOf[(BufferParam[Int, ReadOnly], BufferParam[Int, ReadWrite],
            BufferParam[Int, ReadWrite], ScalarParam[Int], ScalarParam[Int])]
          when(i < p._4) {
            val total = declare(p._1(i).read)
            val loop = pending.dequeue().asInstanceOf[ForLoop]
            gpuFor(loop.index.name, literal(0), p._5, loop.step) { round =>
              val step = declare(round + literal(1))
              val before = save(read(total))
              write(step, read(step) + literal(1))
              write(total, before + read(step))
              write(p._3(i), before)
            }(using summon[BlockBuilder], DslSourcePosition(loop.span))
            write(p._2(i), read(total))
          }(using summon[BlockBuilder], DslSourcePosition(guard.span))
      }
      assert(pending.isEmpty)
      assertEquals(actual.ir, reference.ir)
      assert(KernelValidator.validate(actual).isValid)
      assertEquals(KernelValidator.validate(actual), KernelValidator.validate(reference))
      assertEquals(EffectAnalysis.block(actual.body), EffectAnalysis.block(reference.body))
      assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(reference))

  test("mutable declarations and assignments retain source mapping lexical scope and deterministic names"):
    val factories = Vector(
      (() => PrototypeKernels.mutableBranches, 4),
      (() => PrototypeKernels.mutableLoops, 4),
      (() => PrototypeKernels.mutableRowSums, 3))
    factories.foreach { (factory, expected) =>
      val actual = factory()
      assertEquals(actual.body.statements.size, 2)
      val all = statements(actual.body)
      val locals = all.collect { case declaration: LocalDeclaration[?] => declaration }
      val writes = all.collect { case store: Store[?, ?] if store.to.isInstanceOf[LocalVariable[?]] => store }
      assertEquals(locals.size, expected)
      assert(writes.nonEmpty)
      assertEquals(locals.map(_.local.name).distinct.size, locals.size)
      val generated = CudaCodegen.generate(actual).toOption.get
      val mapped = generated.sourceMap.entries.map(_.sourceSpan)
      (locals ++ writes).foreach { statement =>
        assert(statement.span.file.replace('\\', '/').endsWith("examples/PrototypeKernels.scala"))
        assertNotEquals(statement.span, SourceSpan.Unknown)
        assert(mapped.contains(statement.span))
      }
      assert(KernelValidator.validate(actual).isValid)
      val second = factory()
      assertNotEquals(locals.head.local.name, statements(second.body).head.asInstanceOf[LocalDeclaration[?]].local.name)
      assertEquals(generated.cudaSource, CudaCodegen.generate(second).toOption.get.cudaSource)
    }

  test("nested branches scopes and loops exactly match explicit let IR effects and artifacts"):
    for branches <- Vector(true, false) do
      val actual = if branches then PrototypeKernels.nestedBranches else PrototypeKernels.nestedLoops
      val pending = scala.collection.mutable.Queue.from(statements(actual.body))
      def save(initial: Expr[Int])(using builder: BlockBuilder): Expr[Int] =
        val declaration = pending.dequeue().asInstanceOf[LocalDeclaration[Int]]
        let(declaration.local.name, initial)(using summon[flight4s.core.types.CudaType[Int]], builder,
          DslSourcePosition(declaration.span))
      def write(target: Place[Int, Global, ReadWrite], value: Expr[Int])(using builder: BlockBuilder): Unit =
        val store = pending.dequeue().asInstanceOf[Store[Int, Global]]
        target.:=(value)(using builder, DslSourcePosition(store.span))
      def guarded(condition: Expr[Boolean])(body: BlockBuilder ?=> Unit)(using builder: BlockBuilder): Unit =
        val branch = pending.dequeue().asInstanceOf[IfThen]
        when(condition)(body)(using builder, DslSourcePosition(branch.span))
      def scope(body: BlockBuilder ?=> Unit)(using builder: BlockBuilder): Unit =
        val node = pending.dequeue().asInstanceOf[ScopedBlock]
        scoped(body)(using builder, DslSourcePosition(node.span))
      def alternative(condition: Expr[Boolean])(yes: BlockBuilder ?=> Unit)(no: BlockBuilder ?=> Unit)(using builder: BlockBuilder): Unit =
        val node = pending.dequeue().asInstanceOf[IfThen]
        gpuIf(condition)(yes)(no)(using builder, DslSourcePosition(node.span))
      def loop(from: Expr[Int], until: Expr[Int])(body: Expr[Int] => (BlockBuilder ?=> Unit))(using builder: BlockBuilder): Unit =
        val node = pending.dequeue().asInstanceOf[ForLoop]
        gpuFor(node.index.name, from, until, node.step)(body)(using builder, DslSourcePosition(node.span))
      val reference = CudaDsl.kernel(actual.name, actual.signature) { bindings =>
        val i = save(blockIdx.x * blockDim.x + threadIdx.x)
        if branches then
          val p = bindings.asInstanceOf[(BufferParam[Int, ReadWrite], BufferParam[Int, ReadWrite], BufferParam[Int, ReadWrite], ScalarParam[Int])]
          guarded(i < p._4) {
            val original = save(p._1(i).read)
            scope {
              val alias = save(original)
              alternative((i % literal(2)) === literal(0)) {
                val result = save(alias + literal(1))
                write(p._1(i), literal(900) + i)
                write(p._2(i), result)
                write(p._3(i), p._1(i).read)
              } {
                val result = save(alias - literal(1))
                write(p._1(i), literal(900) + i)
                write(p._2(i), result)
                write(p._3(i), p._1(i).read)
              }
            }
          }
        else
          val p = bindings.asInstanceOf[(BufferParam[Int, ReadWrite], BufferParam[Int, ReadWrite], ScalarParam[Int], ScalarParam[Int])]
          guarded(i < p._3) {
            loop(literal(0), p._4) { round =>
              val original = save(p._1(i).read)
              val alias = save(original)
              write(p._1(i), original + round + literal(1))
              write(p._2(i), alias)
            }
          }
      }
      assert(pending.isEmpty)
      assertEquals(actual.ir, reference.ir)
      assert(KernelValidator.validate(actual).isValid)
      assertEquals(KernelValidator.validate(actual), KernelValidator.validate(reference))
      assertEquals(EffectAnalysis.block(actual.body), EffectAnalysis.block(reference.body))
      assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(reference))

  test("nested snapshot declarations retain lexical placement source spans and deterministic names"):
    val factories = Vector(
      (() => PrototypeKernels.nestedBranches, 5),
      (() => PrototypeKernels.nestedLoops, 3),
      (() => PrototypeKernels.nestedTraversal, 2))
    factories.foreach { (factory, expected) =>
      val definition = factory()
      assertEquals(definition.body.statements.size, 2)
      assert(definition.body.statements.head.isInstanceOf[LocalDeclaration[?]])
      assert(definition.body.statements.last.isInstanceOf[IfThen])
      val declarations = statements(definition.body).collect { case declaration: LocalDeclaration[?] => declaration }
      assertEquals(declarations.size, expected)
      assertEquals(declarations.map(_.local.name).distinct.size, expected)
      val generated = CudaCodegen.generate(definition).toOption.get
      val mapped = generated.sourceMap.entries.map(_.sourceSpan)
      declarations.foreach { declaration =>
        assert(declaration.span.file.replace('\\', '/').endsWith("examples/PrototypeKernels.scala"))
        assertNotEquals(declaration.span, SourceSpan.Unknown)
        assert(mapped.contains(declaration.span))
      }
      val second = factory()
      assertNotEquals(declarations.head.local.name,
        statements(second.body).head.asInstanceOf[LocalDeclaration[?]].local.name)
      assertEquals(generated.cudaSource, CudaCodegen.generate(second).toOption.get.cudaSource)
    }

  test("annotation snapshots exactly match manual let IR effects validation and generated CUDA"):
    val actual = PrototypeKernels.intSnapshots
    val declarations = actual.body.statements.take(3).map(_.asInstanceOf[LocalDeclaration[Int]])
    val stores = actual.body.statements.drop(3).map(_.asInstanceOf[Store[Int, Global]])
    def saved(index: Int, initial: Expr[Int])(using BlockBuilder): Expr[Int] =
      given DslSourcePosition = DslSourcePosition(declarations(index).span)
      let(declarations(index).local.name, initial)
    val reference = CudaDsl.kernel(actual.name, actual.signature) { bindings =>
      // Both definitions share the exact signature, including binding access witnesses.
      val p = bindings.asInstanceOf[(BufferParam[Int, ReadWrite], BufferParam[Int, ReadWrite], BufferParam[Int, ReadWrite])]
      val i = saved(0, blockIdx.x * blockDim.x + threadIdx.x)
      val original = saved(1, p._1(i).read)
      val alias = saved(2, original)
      p._1(i).:=(literal(900) + i)(using summon[BlockBuilder], DslSourcePosition(stores(0).span))
      p._2(i).:=(original + alias)(using summon[BlockBuilder], DslSourcePosition(stores(1).span))
      p._3(i).:=(p._1(i).read)(using summon[BlockBuilder], DslSourcePosition(stores(2).span))
    }
    assertEquals(actual.ir, reference.ir)
    assertEquals(KernelValidator.validate(actual), KernelValidator.validate(reference))
    assert(KernelValidator.validate(actual).isValid)
    assertEquals(EffectAnalysis.block(actual.body), EffectAnalysis.block(reference.body))
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(reference))

  test("factory calls get fresh binding identities and deterministic CUDA names"):
    val first = PrototypeKernels.intSnapshots
    val second = PrototypeKernels.intSnapshots
    val a = first.body.statements.head.asInstanceOf[LocalDeclaration[Int]]
    val b = second.body.statements.head.asInstanceOf[LocalDeclaration[Int]]
    assertNotEquals(a.local.name, b.local.name)
    assertEquals(CudaCodegen.generate(first).toOption.get.cudaSource, CudaCodegen.generate(second).toOption.get.cudaSource)

  test("snapshot spans retain original Scala val lines and generated source mapping"):
    val definition = PrototypeKernels.intSnapshots
    val declarations = definition.body.statements.take(3).map(_.asInstanceOf[LocalDeclaration[Int]])
    val spans = declarations.map(_.span)
    assert(spans.forall(_.file.replace('\\', '/').endsWith("examples/PrototypeKernels.scala")))
    assert(spans.forall(_ != SourceSpan.Unknown))
    assertEquals(spans.map(_.startLine), spans.map(_.startLine).sorted.distinct)
    assert(spans.sliding(2).forall(pair => pair(0).startLine < pair(1).startLine))
    val map = CudaCodegen.generate(definition).toOption.get.sourceMap.entries.map(_.sourceSpan)
    spans.foreach(span => assert(map.contains(span)))

  test("explicit snapshots stay single declarations and work in nested DSL branches"):
    val definition = PrototypeKernels.explicitSnapshot
    assertEquals(definition.body.statements.size, 2)
    val declaration = definition.body.statements.head.asInstanceOf[LocalDeclaration[Int]]
    assertEquals(declaration.local.name, "saved")
    val branch = definition.body.statements.last.asInstanceOf[IfThen]
    assertEquals(branch.thenBlock.statements.size, 2)
    assertEquals(branch.thenBlock.statements.head.asInstanceOf[LocalDeclaration[Int]].local.name, "nested")
    assert(KernelValidator.validate(definition).isValid)

  test("parameter ABI and read-only access survive annotation expansion"):
    val definition = PrototypeKernels.vectorAdd
    assertEquals(definition.params.map(_.name), Vector("left", "right", "target", "count"))
    assertEquals(definition.params.take(3).map(_.asInstanceOf[BufferParam[Float, ?]].access),
      Vector(BufferAccess.ReadOnly, BufferAccess.ReadOnly, BufferAccess.ReadWrite))
    assert(KernelValidator.validate(definition).isValid)
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("const float* left"))
    assert(generated.cudaSource.contains("const float* right"))
