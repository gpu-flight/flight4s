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
    case loop: ForLoop => Vector(loop) ++ all(loop.body)
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
      case (left: LoopIndex, right: LoopIndex) => left.copy(span = right.span)
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
      case (left: ForLoop, right: ForLoop) =>
        left.copy(index = left.index.copy(span = right.index.span), from = align(left.from, right.from),
          until = align(left.until, right.until), body = alignBlock(left.body, right.body), span = right.span)
      case (left: ScopedBlock, right: ScopedBlock) =>
        left.copy(body = alignBlock(left.body, right.body), span = right.span)
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

  private def namesOf(kernel: Kernel[?]): scala.collection.mutable.Queue[String] =
    scala.collection.mutable.Queue.from(all(kernel.body).flatMap {
      case declaration: LocalDeclaration[?] => Vector(declaration.local.name)
      case loop: ForLoop => Vector(loop.index.name)
      case _ => Vector.empty
    })

  private def assertReference[Args <: Tuple](actual: Kernel[Args], reference: Kernel[Args]): Unit =
    val aligned = reference.copy(ir = reference.ir.copy(signature = actual.signature, body = alignBlock(reference.body, actual.body)))
    assertEquals(actual.ir, aligned.ir)
    assertEquals(EffectAnalysis.block(actual.body), EffectAnalysis.block(aligned.body))
    assertEquals(KernelValidator.validate(actual), KernelValidator.validate(aligned))
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(aligned))

  test("quoted row loops exactly match ordered explicit DSL loops and bound snapshots"):
    val actual = ScalaKernels.rowSum
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Float]("data"), output[Float]("target"),
        value[Int]("rows"), value[Int]("columns"))) { p =>
      val rows = declare(p._3).read
      val columns = declare(p._4).read
      scoped {
        val row = declare(blockIdx.x * blockDim.x + threadIdx.x).read
        when(row < rows) {
          val total = declare(literal(0.0f))
          val start = declare(literal(0)).read
          val end = declare(columns).read
          gpuFor(names.dequeue(), start, end) { column =>
            val item = declare(p._1(row * columns + column).read).read
            total := total.read + item
          }
          p._2(row) := total.read
        }
      }
    }
    assert(names.isEmpty, actual.ir.toString)
    assertReference(actual, reference)

  test("quoted mutable range bounds exactly match snapshots rather than live loop bounds"):
    val actual = ScalaKernels.rangeBounds
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(output[Int]("visits"), output[Int]("last"),
        value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val begin = declare(p._4)
        val end = declare(p._5)
        val visits = declare(literal(0))
        val last = declare(literal(123))
        val capturedBegin = declare(begin.read).read
        val capturedEnd = declare(end.read).read
        gpuFor(names.dequeue(), capturedBegin, capturedEnd) { index =>
          val snapshot = declare(index).read
          visits := visits.read + literal(1)
          last := snapshot
          begin := literal(0)
          end := literal(0)
        }
        p._1(lane) := visits.read
        p._2(lane) := last.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("quoted nested ranges exactly match per-iteration snapshots and independent shadowed indices"):
    val actual = ScalaKernels.nestedRanges
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("rounds"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val total = declare(p._1(lane).read)
        val outerStart = declare(literal(0)).read
        val outerEnd = declare(p._4).read
        gpuFor(names.dequeue(), outerStart, outerEnd) { outer =>
          val before = declare(total.read).read
          val innerStart = declare(literal(0)).read
          val innerEnd = declare(outer).read
          gpuFor(names.dequeue(), innerStart, innerEnd) { inner =>
            val previous = declare(total.read).read
            total := previous + before + inner
          }
        }
        p._2(lane) := total.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("quoted range guards exactly match lazy nested branches without speculative loads"):
    val actual = ScalaKernels.guardedRows
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Float]("data"), output[Float]("target"),
        value[Int]("rows"), value[Int]("columns"), value[Float]("threshold"))) { p =>
      val row = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(row < p._3) {
        val total = declare(literal(0.0f))
        val start = declare(literal(-1)).read
        val end = declare(p._4 + literal(1)).read
        gpuFor(names.dequeue(), start, end) { column =>
          when(column >= literal(0)) {
            when(column < p._4) {
              when(p._1(row * p._4 + column).read > p._5) {
                val item = declare(p._1(row * p._4 + column).read).read
                total := total.read + item
              }
            }
          }
        }
        p._2(row) := total.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("quoted withFilter guards read live state while loop bounds remain captured"):
    val actual = ScalaKernels.guardedState
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val total = declare(p._1(lane).read)
        val begin = declare(p._4)
        val end = declare(p._5)
        val capturedBegin = declare(begin.read).read
        val capturedEnd = declare(end.read).read
        gpuFor(names.dequeue(), capturedBegin, capturedEnd) { index =>
          when(index >= total.read) {
            when(index % literal(2) === literal(0)) {
              val before = declare(total.read).read
              total := before + index + literal(1)
              begin := literal(0)
              end := literal(0)
            }
          }
        }
        p._2(lane) := total.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("quoted guarded nested generators and index shadows exactly match lexical loop structure"):
    val actual = ScalaKernels.nestedGuards
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("rounds"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val total = declare(p._1(lane).read)
        val start = declare(literal(0)).read
        val end = declare(p._4).read
        gpuFor(names.dequeue(), start, end) { outer =>
          when(outer % literal(2) === literal(0)) {
            val innerStart = declare(literal(-1)).read
            val innerEnd = declare(outer + literal(1)).read
            gpuFor(names.dequeue(), innerStart, innerEnd) { inner =>
              when(inner >= literal(0)) {
                when(inner < outer) {
                  val before = declare(total.read).read
                  total := before + outer + inner
                }
              }
            }
          }
        }
        val secondStart = declare(literal(0)).read
        val secondEnd = declare(p._4).read
        gpuFor(names.dequeue(), secondStart, secondEnd) { outer =>
          when(outer > literal(0)) {
            val innerStart = declare(literal(0)).read
            val innerEnd = declare(outer).read
            gpuFor(names.dequeue(), innerStart, innerEnd) { inner =>
              when(inner % literal(2) === literal(0)) { total := total.read + inner }
            }
          }
        }
        p._2(lane) := total.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("all fixtures preserve source maps scope unique names validation and deterministic CUDA"):
    val factories: Vector[() => Kernel[?]] = Vector(() => ScalaKernels.scale, () => ScalaKernels.branches,
      () => ScalaKernels.shortCircuit, () => ScalaKernels.doubles, () => ScalaKernels.rowSum,
      () => ScalaKernels.rangeBounds, () => ScalaKernels.nestedRanges, () => ScalaKernels.guardedRows,
      () => ScalaKernels.guardedState, () => ScalaKernels.nestedGuards)
    factories.foreach { factory =>
      val actual = factory()
      val statements = all(actual.body)
      val locals = statements.collect { case d: LocalDeclaration[?] => d.local }
      assertEquals(locals.map(_.name).distinct.size, locals.size)
      val indices = statements.collect { case loop: ForLoop => loop.index.name }
      assertEquals((locals.map(_.name) ++ indices).distinct.size, locals.size + indices.size)
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
