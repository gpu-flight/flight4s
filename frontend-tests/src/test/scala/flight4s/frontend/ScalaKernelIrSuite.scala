package flight4s.core.frontend

import scala.annotation.experimental
import munit.FunSuite
import flight4s.frontend.examples.ScalaKernels
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.{CudaDsl, DslSourcePosition}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.CudaType

@experimental
class ScalaKernelIrSuite extends FunSuite:
  private def all(block: Block): Vector[Stmt] = block.statements.flatMap {
    case branch: IfThen => Vector(branch) ++ all(branch.thenBlock) ++ branch.elseBlock.toVector.flatMap(all)
    case scope: ScopedBlock => Vector(scope) ++ all(scope.body)
    case statement => Vector(statement)
  }

  // Reference code has different locations. Only spans are aligned, never operations or values.
  private def alignPlace[T, S <: AddressSpace, M <: AccessMode](expected: Place[T, S, M],
      actual: Place[?, ?, ?]): Place[T, S, M] =
    val result: Place[?, ?, ?] = (expected, actual) match
      case (left: LocalVariable[?], right: LocalVariable[?]) => left.copy(span = right.span)
      case (left: BufferElement[?, ?], right: BufferElement[?, ?]) =>
        left.copy(index = align(left.index, right.index), span = right.span)
      case _ => fail(s"place shapes differ: $expected / $actual")
    result.asInstanceOf[Place[T, S, M]]

  private def align[T](expected: Expr[T], actual: Expr[?]): Expr[T] =
    val result: Expr[?] = (expected, actual) match
      case (left: Literal[?], right: Literal[?]) => left.copy(span = right.span)
      case (left: Intrinsic[?], right: Intrinsic[?]) => left.copy(span = right.span)
      case (left: ScalarParam[?], _: ScalarParam[?]) => left
      case (left: Load[?, ?, ?], right: Load[?, ?, ?]) =>
        left.copy(from = alignPlace(left.from, right.from), span = right.span)
      case (left: Binary[?], right: Binary[?]) =>
        left.copy(left = align(left.left, right.left), right = align(left.right, right.right), span = right.span)
      case (left: Compare[?], right: Compare[?]) =>
        left.copy(left = align(left.left, right.left), right = align(left.right, right.right), span = right.span)
      case (left: Conditional[?], right: Conditional[?]) =>
        left.copy(condition = align(left.condition, right.condition), whenTrue = align(left.whenTrue, right.whenTrue),
          whenFalse = align(left.whenFalse, right.whenFalse), span = right.span)
      case _ => fail(s"expression shapes differ: $expected / $actual")
    result.asInstanceOf[Expr[T]]

  private def alignBlock(expected: Block, actual: Block): Block =
    assertEquals(expected.statements.size, actual.statements.size)
    Block(expected.statements.zip(actual.statements).map { (left, right) => (left, right) match
      case (left: LocalDeclaration[?], right: LocalDeclaration[?]) =>
        left.copy(local = left.local.copy(span = right.local.span), initial = align(left.initial, right.initial), span = right.span)
      case (left: Store[?, ?], right: Store[?, ?]) =>
        left.copy(to = alignPlace(left.to, right.to), value = align(left.value, right.value), span = right.span)
      case (left: IfThen, right: IfThen) =>
        assertEquals(left.elseBlock.isDefined, right.elseBlock.isDefined)
        left.copy(condition = align(left.condition, right.condition), thenBlock = alignBlock(left.thenBlock, right.thenBlock),
          elseBlock = left.elseBlock.zip(right.elseBlock).map(alignBlock), span = right.span)
      case _ => fail(s"statement shapes differ: $left / $right")
    })

  test("Scala branches snapshots shadowing and conditional expressions exactly match explicit DSL artifacts"):
    val actual = ScalaKernels.branches
    val names = scala.collection.mutable.Queue.from(all(actual.body).collect { case d: LocalDeclaration[?] => d.local.name })
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] =
      local(names.dequeue(), initial)(using summon[CudaType[T]], summon[BlockBuilder], DslSourcePosition(SourceSpan.Unknown))
    val reference = CudaDsl.kernel(actual.name, params(inOut[Int]("data"), output[Int]("saved"),
        output[Int]("target"), value[Int]("count"))) { p =>
      val i = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(i < p._4) {
        val total = declare(p._1(i).read)
        val before = declare(total.read).read
        gpuIf(i % literal(2) === literal(0)) { total := total.read + literal(3) } { total := total.read - literal(2) }
        p._1(i) := total.read
        p._2(i) := before
        gpuIf(before < literal(0)) {
          val inner = declare(before + literal(7)).read
          p._3(i) := inner
        } {
          p._3(i) := choose(total.read > literal(0))(total.read)(before)
        }
      }
    }
    assert(names.isEmpty)
    val aligned = reference.copy(ir = reference.ir.copy(signature = actual.signature, body = alignBlock(reference.body, actual.body)))
    assertEquals(actual.ir, aligned.ir)
    assertEquals(EffectAnalysis.block(actual.body), EffectAnalysis.block(aligned.body))
    assertEquals(KernelValidator.validate(actual), KernelValidator.validate(aligned))
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(aligned))

  test("short-circuit boolean expressions retain lazy Conditional IR and exactly match explicit DSL"):
    val actual = ScalaKernels.shortCircuit
    val names = scala.collection.mutable.Queue.from(all(actual.body).collect { case d: LocalDeclaration[?] => d.local.name })
    val reference = CudaDsl.kernel(actual.name, params(input[Int]("data"), output[Int]("target"), value[Int]("count"))) { p =>
      val i = let(names.dequeue(), blockIdx.x * blockDim.x + threadIdx.x)
      when((i < p._3) && ((p._1(i).read !== literal(0)) || (p._1(i).read === literal(0)))) {
        val nonzero = let(names.dequeue(), p._1(i).read !== literal(0))
        p._2(i) := choose(!nonzero)(literal(11))(p._1(i).read + literal(2))
      }
    }
    assert(names.isEmpty)
    val aligned = reference.copy(ir = reference.ir.copy(signature = actual.signature, body = alignBlock(reference.body, actual.body)))
    assertEquals(actual.ir, aligned.ir)
    assertEquals(EffectAnalysis.block(actual.body), EffectAnalysis.block(aligned.body))
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(aligned))

  test("all fixtures preserve source maps scope unique names validation and deterministic CUDA"):
    val factories: Vector[() => Kernel[?]] = Vector(() => ScalaKernels.scale, () => ScalaKernels.branches,
      () => ScalaKernels.shortCircuit, () => ScalaKernels.doubles)
    factories.foreach { factory =>
      val actual = factory()
      val statements = all(actual.body)
      val locals = statements.collect { case d: LocalDeclaration[?] => d.local }
      assertEquals(locals.map(_.name).distinct.size, locals.size)
      assert(KernelValidator.validate(actual).isValid)
      val generated = CudaCodegen.generate(actual).toOption.get
      val mapped = generated.sourceMap.entries.map(_.sourceSpan)
      statements.foreach { statement =>
        assertNotEquals(statement.span, SourceSpan.Unknown)
        assert(statement.span.file.replace('\\', '/').endsWith("examples/ScalaKernels.scala"))
        assert(mapped.contains(statement.span), s"unmapped statement: $statement")
      }
      val second = factory()
      assertNotEquals(locals.head.name, all(second.body).collectFirst { case d: LocalDeclaration[?] => d.local.name }.get)
      assertEquals(generated.cudaSource, CudaCodegen.generate(second).toOption.get.cudaSource)
    }
