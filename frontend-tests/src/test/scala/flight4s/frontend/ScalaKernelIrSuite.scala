package flight4s.core.frontend

import scala.annotation.experimental
import munit.FunSuite
import flight4s.frontend.examples.ScalaKernels
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.{CudaDsl, DslSourcePosition}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.CudaType
import flight4s.core.launch.{Block as LaunchBlock}

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
      case (left: SharedElement[?], right: SharedElement[?]) =>
        assertEquals(left.indices.size, right.indices.size)
        left.copy(indices = left.indices.zip(right.indices).map(align), span = right.span)
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
      case (left: Barrier, right: Barrier) => left.copy(span = right.span)
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
    assertEquals(reference.sharedMemory.size, actual.sharedMemory.size)
    val memory = reference.sharedMemory.zip(actual.sharedMemory).map { (left, right) =>
      left.copy(span = right.span)(using left.rankWitness)
    }
    val aligned = reference.copy(ir = reference.ir.copy(signature = actual.signature,
      sharedMemory = memory, body = alignBlock(reference.body, actual.body)))
    assertEquals(actual.ir, aligned.ir)
    assertEquals(EffectAnalysis.block(actual.body), EffectAnalysis.block(aligned.body))
    assertEquals(KernelValidator.validate(actual), KernelValidator.validate(aligned))
    assertEquals(CudaCodegen.generate(actual), CudaCodegen.generate(aligned))

  test("shared exchange exactly matches explicit shared declarations stores loads barriers and effects"):
    val actual = ScalaKernels.sharedExchange
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): Expr[T] = local(names.dequeue(), initial).read
    val reference = CudaDsl.kernel(actual.name, params(input[Float]("data"), output[Float]("target"), value[Int]("count"))) { p =>
      val tile = sharedArray[Float](actual.sharedMemory.head.name, 64)
      val count = declare(p._3)
      scoped {
        val lane = declare(threadIdx.x)
        val i = declare(blockIdx.x * blockDim.x + lane)
        tile(lane) := choose(i < count)(p._1(i).read)(literal(0.0f))
        barrier()
        when(i < count) { p._2(i) := tile((lane + literal(1)) % literal(64)).read }
      }
    }.requiringBlock(LaunchBlock.x(64))
    assert(names.isEmpty)
    assertReference(actual, reference)
    val effects = EffectAnalysis.block(actual.body)
    assert(effects.readSpaces.contains(EffectMemorySpace.Shared))
    assert(effects.writtenSpaces.contains(EffectMemorySpace.Shared))
    assert(effects.hasBarrier)
    assertEquals(KernelValidator.validate(actual).warnings, Vector.empty)

  test("shared reuse exactly matches serial loops and explicit read-before-overwrite barriers"):
    val actual = ScalaKernels.sharedReuse
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): Expr[T] = local(names.dequeue(), initial).read
    val reference = CudaDsl.kernel(actual.name, params(input[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("rounds"))) { p =>
      val tile = sharedArray[Int](actual.sharedMemory.head.name, 64)
      val lane = declare(threadIdx.x)
      val i = declare(blockIdx.x * blockDim.x + lane)
      tile(lane) := choose(i < p._3)(p._1(i).read)(literal(0))
      barrier()
      val start = declare(literal(0))
      val end = declare(p._4)
      gpuFor(names.dequeue(), start, end) { round =>
        val previous = declare(tile((lane + literal(1)) % literal(64)).read)
        barrier()
        tile(lane) := previous + round
        barrier()
      }
      when(i < p._3) { p._2(i) := tile(lane).read }
    }.requiringBlock(LaunchBlock.x(64))
    assert(names.isEmpty)
    assertReference(actual, reference)
    assertEquals(KernelValidator.validate(actual).warnings, Vector.empty)

  test("Boolean and Double shared arrays preserve lazy reads and distinct storage"):
    val actual = ScalaKernels.sharedFlags
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): Expr[T] = local(names.dequeue(), initial).read
    val reference = CudaDsl.kernel(actual.name, params(input[Double]("data"), output[Double]("target"),
        value[Int]("count"), value[Boolean]("enabled"))) { p =>
      val flags = sharedArray[Boolean](actual.sharedMemory(0).name, 64)
      val values = sharedArray[Double](actual.sharedMemory(1).name, 64)
      val count = declare(p._3)
      val enabled = declare(p._4)
      scoped {
        val lane = declare(threadIdx.x)
        val i = declare(blockIdx.x * blockDim.x + lane)
        flags(lane) := enabled && i < count
        values(lane) := choose(flags(lane).read)(p._1(i).read)(literal(0.0))
        barrier()
        val neighbor = declare((lane + literal(1)) % literal(64))
        when(i < count) { p._2(i) := choose(flags(neighbor).read)(values(neighbor).read)(literal(-7.0)) }
      }
    }.requiringBlock(LaunchBlock.x(64))
    assert(names.isEmpty)
    assertReference(actual, reference)
    assertEquals(KernelValidator.validate(actual).warnings, Vector.empty)

  test("phase exchange exactly matches scoped explicit DSL statements and one trailing barrier"):
    val actual = ScalaKernels.phaseExchange
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): Expr[T] = local(names.dequeue(), initial).read
    val reference = CudaDsl.kernel(actual.name, params(input[Float]("data"), output[Float]("target"), value[Int]("count"))) { p =>
      val tile = sharedArray[Float](actual.sharedMemory.head.name, 64)
      val count = declare(p._3)
      scoped {
        val lane = declare(threadIdx.x)
        val i = declare(blockIdx.x * blockDim.x + lane)
        scoped {
          val item = declare(choose(i < count)(p._1(i).read)(literal(0.0f)))
          tile(lane) := item
        }
        barrier()
        when(i < count) { p._2(i) := tile((lane + literal(1)) % literal(64)).read }
      }
    }.requiringBlock(LaunchBlock.x(64))
    assert(names.isEmpty)
    assertReference(actual, reference)
    assertEquals(KernelValidator.validate(actual).warnings, Vector.empty)

  test("repeated phases keep explicit read-before-overwrite barriers and one trailing barrier per round"):
    val actual = ScalaKernels.phaseReuse
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): Expr[T] = local(names.dequeue(), initial).read
    val reference = CudaDsl.kernel(actual.name, params(input[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("rounds"))) { p =>
      val tile = sharedArray[Int](actual.sharedMemory.head.name, 64)
      val lane = declare(threadIdx.x)
      val i = declare(blockIdx.x * blockDim.x + lane)
      scoped { tile(lane) := choose(i < p._3)(p._1(i).read)(literal(0)) }
      barrier()
      val start = declare(literal(0))
      val end = declare(p._4)
      gpuFor(names.dequeue(), start, end) { round =>
        scoped {
          val previous = declare(tile((lane + literal(1)) % literal(64)).read)
          barrier()
          tile(lane) := previous + round
        }
        barrier()
      }
      when(i < p._3) { p._2(i) := tile(lane).read }
    }.requiringBlock(LaunchBlock.x(64))
    assert(names.isEmpty)
    assertReference(actual, reference)
    assertEquals(KernelValidator.validate(actual).warnings, Vector.empty)

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

  test("staged yield guards and map snapshots exactly match independent ordered DSL loops"):
    val actual = ScalaKernels.yieldRows
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
              val item = declare(p._1(row * p._4 + column).read).read
              when(item > p._5) {
                val doubled = declare(item * literal(2.0f)).read
                total := total.read + doubled
              }
            }
          }
        }
        p._2(row) := total.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("reused yield plans retain construction bounds and snapshot each mapped item before stores"):
    val actual = ScalaKernels.yieldReuse
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(inOut[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val begin = declare(p._4)
        val end = declare(p._5)
        val bias = declare(literal(0))
        val total = declare(literal(0))
        val capturedStart = declare(begin.read).read
        val capturedEnd = declare(end.read).read
        begin := literal(0)
        end := literal(0)
        bias := literal(1)
        def visit()(using BlockBuilder): Unit =
          gpuFor(names.dequeue(), capturedStart, capturedEnd) { index =>
            when(index % literal(2) === literal(0)) {
              val item = declare(p._1(lane).read + index + bias.read).read
              p._1(lane) := item + literal(1)
              total := total.read + (item + item)
            }
          }
        visit()
        bias := literal(3)
        visit()
        p._2(lane) := total.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("nested traversal plans preserve lexical map captures live guards and terminal shadows"):
    val actual = ScalaKernels.yieldNested
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
          val before = declare(total.read).read
          val innerStart = declare(literal(0)).read
          val innerEnd = declare(outer).read
          gpuFor(names.dequeue(), innerStart, innerEnd) { index =>
            val item = declare(index + before).read
            when(item >= total.read) {
              val mapped = declare(item + outer).read
              total := total.read + mapped
            }
          }
        }
        p._2(lane) := total.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("ordered guarded Float folds exactly match seed snapshots and serial updates"):
    val actual = ScalaKernels.foldRows
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Float]("data"), output[Float]("target"),
        value[Int]("rows"), value[Int]("columns"), value[Float]("threshold"), value[Float]("seed"))) { p =>
      val row = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(row < p._3) {
        val start = declare(literal(-1)).read
        val end = declare(p._4 + literal(1)).read
        val accumulator = declare(p._6)
        gpuFor(names.dequeue(), start, end) { column =>
          when(column >= literal(0)) {
            when(column < p._4) {
              val item = declare(p._1(row * p._4 + column).read).read
              when(item > p._5) {
                val doubled = declare(item * literal(2.0f)).read
                accumulator := accumulator.read - doubled
              }
            }
          }
        }
        val total = declare(accumulator.read).read
        p._2(row) := total
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("reused ordered folds independently capture seeds live maps and saved earlier results"):
    val actual = ScalaKernels.foldReuse
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(inOut[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val begin = declare(p._4)
        val end = declare(p._5)
        val bias = declare(literal(0))
        val capturedStart = declare(begin.read).read
        val capturedEnd = declare(end.read).read
        begin := literal(0)
        end := literal(0)
        bias := literal(1)
        val firstState = declare(p._1(lane).read)
        gpuFor(names.dequeue(), capturedStart, capturedEnd) { index =>
          when(index % literal(2) === literal(0)) {
            val item = declare(p._1(lane).read + index + bias.read).read
            firstState := firstState.read - item - item
          }
        }
        val first = declare(firstState.read).read
        p._1(lane) := first
        bias := literal(3)
        val secondState = declare(first + literal(1))
        gpuFor(names.dequeue(), capturedStart, capturedEnd) { index =>
          when(index % literal(2) === literal(0)) {
            val item = declare(p._1(lane).read + index + bias.read).read
            secondState := choose(item > secondState.read)(item)(secondState.read - item)
          }
        }
        val second = declare(secondState.read)
        second := second.read + first
        p._2(lane) := second.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("nested Boolean and Double folds preserve independent primitive states lexical shadows and refresh"):
    val actual = ScalaKernels.foldNested
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Double]("data"), output[Double]("target"),
        value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val total = declare(p._1(lane).read)
        val start = declare(literal(0)).read
        val end = declare(p._4).read
        gpuFor(names.dequeue(), start, end) { outer =>
          val before = declare(total.read).read
          val innerStart = declare(literal(0)).read
          val innerEnd = declare(outer).read
          val acceptedState = declare(p._5)
          gpuFor(names.dequeue(), innerStart, innerEnd) { index =>
            val item = declare(index + literal(1)).read
            acceptedState := acceptedState.read || (item % literal(2) === literal(0))
          }
          val accepted = declare(acceptedState.read).read
          val nextState = declare(before)
          gpuFor(names.dequeue(), innerStart, innerEnd) { index =>
            val item = declare(index + literal(1)).read
            nextState := choose(accepted)(nextState.read / literal(2.0))(nextState.read - literal(1.0))
          }
          val next = declare(nextState.read)
          next := next.read + literal(0.25)
          total := next.read
        }
        p._2(lane) := total.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("flatMap inner bounds and loads remain inside outer guards with one ordered Float state"):
    val actual = ScalaKernels.flatMapRows
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Float]("data"), input[Int]("lengths"),
        output[Float]("target"), value[Int]("rows"), value[Int]("columns"),
        value[Float]("threshold"))) { p =>
      val row = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(row < p._4) {
        val start = declare(literal(-1)).read
        val end = declare(p._5 + literal(1)).read
        val state = declare(literal(7.0f))
        gpuFor(names.dequeue(), start, end) { column =>
          when(column >= literal(0)) {
            when(column < p._5) {
              val innerStart = declare(literal(0)).read
              val innerEnd = declare(p._2(column).read).read
              gpuFor(names.dequeue(), innerStart, innerEnd) { inner =>
                when(inner % literal(2) === literal(0)) {
                  val item = declare(p._1(row * p._5 + column).read +
                    choose(inner === literal(0))(literal(0.0f))(literal(0.5f))).read
                  when(item > p._6) {
                    val doubled = declare(item * literal(2.0f)).read
                    state := state.read - doubled
                  }
                }
              }
            }
          }
        }
        val total = declare(state.read).read
        p._3(row) := total
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("reused flatMap snapshots outer values and each inner bound while retaining live terminal captures"):
    val actual = ScalaKernels.flatMapReuse
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(inOut[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val begin = declare(p._4)
        val end = declare(p._5)
        val limit = declare(literal(3))
        val total = declare(literal(0))
        val start = declare(begin.read).read
        val stop = declare(end.read).read
        begin := literal(0)
        end := literal(0)
        gpuFor(names.dequeue(), start, stop) { outer =>
          when(outer % literal(2) === literal(0)) {
            val mapped = declare(p._1(lane).read + outer).read
            val innerStart = declare(literal(0)).read
            val innerEnd = declare(limit.read).read
            gpuFor(names.dequeue(), innerStart, innerEnd) { inner =>
              val item = declare(mapped + inner).read
              when(item >= literal(0)) {
                p._1(lane) := item + literal(1)
                total := total.read + (item + item)
                limit := literal(1)
              }
            }
          }
        }
        limit := literal(2)
        val state = declare(total.read)
        gpuFor(names.dequeue(), start, stop) { outer =>
          when(outer % literal(2) === literal(0)) {
            val mapped = declare(p._1(lane).read + outer).read
            val innerStart = declare(literal(0)).read
            val innerEnd = declare(limit.read).read
            gpuFor(names.dequeue(), innerStart, innerEnd) { inner =>
              val item = declare(mapped + inner).read
              when(item >= literal(0)) { state := state.read - item }
            }
          }
        }
        val result = declare(state.read).read
        p._2(lane) := result
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("chained flatMap reuses captured inner plans and keeps nested shadows and Boolean Double states"):
    val actual = ScalaKernels.flatMapNested
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Double]("data"), output[Double]("target"),
        value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val start = declare(literal(0)).read
        val end = declare(p._4).read
        val fixedStart = declare(literal(0)).read
        val fixedEnd = declare(literal(2)).read
        def visit(consume: Expr[Boolean] => BlockBuilder ?=> Unit)(using BlockBuilder): Unit =
          gpuFor(names.dequeue(), start, end) { outer =>
            gpuFor(names.dequeue(), fixedStart, fixedEnd) { inner =>
              val mapped = declare(outer + inner).read
              val innerStart = declare(literal(0)).read
              val innerEnd = declare(mapped).read
              gpuFor(names.dequeue(), innerStart, innerEnd) { index =>
                val item = declare(index + literal(1)).read
                val flag = declare(choose(item % literal(2) === literal(0))(p._5)(!p._5)).read
                consume(flag)
              }
            }
          }
        val foundState = declare(literal(false))
        visit(flag => foundState := foundState.read || flag)
        val found = declare(foundState.read).read
        val totalState = declare(p._1(lane).read)
        visit { flag =>
          val mapped = declare(choose(flag)(literal(2.0))(literal(1.0))).read
          totalState := totalState.read / literal(2.0) - mapped
        }
        val total = declare(totalState.read).read
        p._2(lane) := choose(found)(total)(p._1(lane).read)
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("seven parameter tuples exactly match explicit DSL IR effects and CUDA artifacts"):
    val actual = ScalaKernels.tupleScale
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): Expr[T] = local(names.dequeue(), initial).read
    val reference = CudaDsl.kernel(actual.name, paramsTuple((input[Float]("data"), output[Float]("target"),
        value[Int]("count"), value[Float]("factor"), value[Float]("bias"),
        value[Boolean]("enabled"), value[Double]("cutoff")))) { p =>
      val count = declare(p._3)
      val factor = declare(p._4)
      val bias = declare(p._5)
      val enabled = declare(p._6)
      val cutoff = declare(p._7)
      scoped {
        val i = declare(blockIdx.x * blockDim.x + threadIdx.x)
        when(i < count && enabled && cutoff > literal(0.0)) {
          p._2(i) := p._1(i).read * factor + bias
        }
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)
    assertEquals(actual.params, reference.params)
    assertEquals(actual.signature.abiDescriptors, reference.signature.abiDescriptors)

  test("tuple row fields guards and ordered scalar folds exactly match primitive explicit IR"):
    val actual = ScalaKernels.tupleRows
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Float]("data"), output[Float]("target"),
        value[Int]("rows"), value[Int]("columns"), value[Float]("threshold"), value[Float]("initial"))) { p =>
      val row = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(row < p._3) {
        val start = declare(literal(-1)).read
        val end = declare(p._4 + literal(1)).read
        val state = declare(p._6)
        gpuFor(names.dequeue(), start, end) { column =>
          when(column >= literal(0)) {
            when(column < p._4) {
              val index = declare(column).read
              val item = declare(p._1(row * p._4 + column).read).read
              val even = declare(column % literal(2) === literal(0)).read
              when(even && item > p._5) {
                val scaled = declare(item * literal(2.0f)).read
                val copiedIndex = declare(index).read
                state := state.read - scaled - choose(copiedIndex === literal(0))(literal(0.5f))(literal(0.0f))
              }
            }
          }
        }
        val result = declare(state.read).read
        p._2(row) := result
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("tuple reuse snapshots every field before terminal stores and recaptures only mapping values"):
    assertTupleReuse(ScalaKernels.tupleReuse)

  test("named tuple reuse has identical primitive snapshot and store semantics"):
    assertTupleReuse(ScalaKernels.namedReuse)

  private def assertTupleReuse(actual: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int, Int, Int)]): Unit =
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(inOut[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val begin = declare(p._4)
        val end = declare(p._5)
        val total = declare(literal(0))
        val start = declare(begin.read).read
        val limit = declare(end.read).read
        begin := literal(0)
        end := literal(0)
        gpuFor(names.dequeue(), start, limit) { i =>
          val first = declare(p._1(lane).read + i).read
          val saved = declare(p._1(lane).read).read
          val index = declare(i).read
          when(first >= literal(0)) {
            p._1(lane) := first + literal(1)
            total := total.read + (saved + saved + index)
          }
        }
        val state = declare(total.read)
        gpuFor(names.dequeue(), start, limit) { i =>
          val first = declare(p._1(lane).read + i).read
          val saved = declare(p._1(lane).read).read
          val index = declare(i).read
          when(first >= literal(0)) { state := state.read - first - saved }
        }
        val result = declare(state.read).read
        p._2(lane) := result
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("tuple flatMap keeps outer snapshots inner bounds identity fields and one scalar state"):
    val actual = ScalaKernels.tupleNested
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Double]("data"), output[Double]("target"),
        value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val fixedStart = declare(literal(0)).read
        val fixedEnd = declare(literal(2)).read
        val start = declare(literal(0)).read
        val end = declare(p._4).read
        val state = declare(p._1(lane).read)
        gpuFor(names.dequeue(), start, end) { outer =>
          val index = declare(outer).read
          val original = declare(p._1(lane).read).read
          val enabled = declare(p._5).read
          val innerStart = declare(literal(0)).read
          val innerEnd = declare(index).read
          gpuFor(names.dequeue(), innerStart, innerEnd) { inner =>
            when(inner % literal(2) === literal(0)) {
              val item = declare(original / literal(2.0)).read
              val flag = declare(enabled).read
              val combined = declare(inner + index).read
              when(flag && combined > literal(0)) {
                val copiedItem = declare(item).read
                val copiedFlag = declare(flag).read
                val copiedIndex = declare(combined).read
                gpuFor(names.dequeue(), fixedStart, fixedEnd) { k =>
                  val value = declare(choose(k === literal(0))(copiedItem)(copiedItem + literal(1.0))).read
                  state := state.read / literal(2.0) - value
                }
              }
            }
          }
        }
        val result = declare(state.read).read
        p._2(lane) := result
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("tuple Float and Int state snapshots all next fields before stores"):
    assertTupleFoldRows(ScalaKernels.tupleFoldRows)

  test("named Float Int folds match independent primitive locals and stores"):
    assertTupleFoldRows(ScalaKernels.namedFoldRows)

  private def assertTupleFoldRows(actual: Kernel[(DeviceBuffer[Float], DeviceBuffer[Float], DeviceBuffer[Int], Int, Int, Float)]): Unit =
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Float]("data"), output[Float]("target"),
        output[Int]("visits"), value[Int]("rows"), value[Int]("columns"), value[Float]("seed"))) { p =>
      val row = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(row < p._4) {
        val start = declare(literal(-1)).read
        val end = declare(p._5 + literal(1)).read
        val total = declare(p._6)
        val visits = declare(literal(0))
        gpuFor(names.dequeue(), start, end) { column =>
          when(column >= literal(0)) {
            when(column < p._5) {
              val index = declare(column).read
              val item = declare(p._1(row * p._5 + column).read).read
              when((index % literal(2) === literal(0)) && (item !== literal(0.0f))) {
                val nextTotal = declare(total.read - item - choose(visits.read % literal(2) === literal(0))(literal(0.5f))(literal(0.0f))).read
                val nextVisits = declare(visits.read + literal(1)).read
                total := nextTotal
                visits := nextVisits
              }
            }
          }
        }
        p._2(row) := total.read
        p._3(row) := visits.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("tuple fold reuse keeps saved results independent of later seeds and cross field updates"):
    assertTupleFoldReuse(ScalaKernels.tupleFoldReuse)

  test("named fold reuse preserves previous fields and independent seeds"):
    assertTupleFoldReuse(ScalaKernels.namedFoldReuse)

  private def assertTupleFoldReuse(actual: Kernel[(DeviceBuffer[Int], DeviceBuffer[Int], Int, Int, Int)]): Unit =
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(inOut[Int]("data"), output[Int]("target"),
        value[Int]("count"), value[Int]("from"), value[Int]("until"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val begin = declare(p._4)
        val end = declare(p._5)
        val start = declare(begin.read).read
        val limit = declare(end.read).read
        begin := literal(0)
        end := literal(0)
        val first = declare(p._1(lane).read)
        val second = declare(p._1(lane).read + literal(1))
        gpuFor(names.dequeue(), start, limit) { i =>
          when(i % literal(2) === literal(0)) {
            val item = declare(p._1(lane).read + i).read
            val nextFirst = declare(second.read).read
            val nextSecond = declare(first.read - item).read
            first := nextFirst
            second := nextSecond
          }
        }
        p._1(lane) := first.read
        val laterFirst = declare(first.read)
        val laterSecond = declare(second.read)
        gpuFor(names.dequeue(), start, limit) { i =>
          when(i % literal(2) === literal(0)) {
            val item = declare(p._1(lane).read + i).read
            val nextFirst = declare(laterSecond.read + item).read
            val nextSecond = declare(laterFirst.read).read
            laterFirst := nextFirst
            laterSecond := nextSecond
          }
        }
        p._1(lane) := first.read + laterSecond.read
        p._2(lane) := second.read + laterFirst.read
      }
    }
    assert(names.isEmpty)
    assertReference(actual, reference)

  test("nested tuple folds refresh Double Boolean seeds and share state across inner traversals"):
    assertTupleFoldNested(ScalaKernels.tupleFoldNested)

  test("nested named folds preserve Double Boolean state across flattening"):
    assertTupleFoldNested(ScalaKernels.namedFoldNested)

  private def assertTupleFoldNested(actual: Kernel[(DeviceBuffer[Double], DeviceBuffer[Double], Int, Int, Boolean)]): Unit =
    val names = namesOf(actual)
    def declare[T](initial: Expr[T])(using CudaType[T], BlockBuilder): LocalVariable[T] = local(names.dequeue(), initial)
    val reference = CudaDsl.kernel(actual.name, params(input[Double]("data"), output[Double]("target"),
        value[Int]("count"), value[Int]("rounds"), value[Boolean]("enabled"))) { p =>
      val lane = declare(blockIdx.x * blockDim.x + threadIdx.x).read
      when(lane < p._3) {
        val total = declare(p._1(lane).read)
        val roundsStart = declare(literal(0)).read
        val roundsEnd = declare(p._4).read
        gpuFor(names.dequeue(), roundsStart, roundsEnd) { round =>
          val start = declare(literal(0)).read
          val end = declare(round).read
          val state = declare(total.read)
          val flag = declare(literal(false))
          gpuFor(names.dequeue(), start, end) { i =>
            val index = declare(i).read
            val item = declare(p._1(lane).read).read
            val enabled = declare(p._5).read
            val innerStart = declare(literal(0)).read
            val innerEnd = declare(index).read
            gpuFor(names.dequeue(), innerStart, innerEnd) { j =>
              val copiedIndex = declare(j).read
              val copiedItem = declare(item).read
              val copiedEnabled = declare(enabled).read
              when(copiedEnabled) {
                val nextState = declare(choose(flag.read)(state.read / literal(2.0) - copiedItem)(state.read - copiedItem)).read
                val nextFlag = declare((state.read > literal(0.0)) !== flag.read).read
                state := nextState
                flag := nextFlag
              }
            }
          }
          total := state.read + choose(flag.read)(literal(0.25))(literal(0.5))
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
      () => ScalaKernels.guardedState, () => ScalaKernels.nestedGuards, () => ScalaKernels.yieldRows,
      () => ScalaKernels.yieldReuse, () => ScalaKernels.yieldNested, () => ScalaKernels.foldRows,
      () => ScalaKernels.foldReuse, () => ScalaKernels.foldNested, () => ScalaKernels.flatMapRows,
      () => ScalaKernels.flatMapReuse, () => ScalaKernels.flatMapNested, () => ScalaKernels.tupleScale,
      () => ScalaKernels.tupleRows, () => ScalaKernels.tupleReuse, () => ScalaKernels.tupleNested,
      () => ScalaKernels.tupleFoldRows, () => ScalaKernels.tupleFoldReuse, () => ScalaKernels.tupleFoldNested,
      () => ScalaKernels.namedFoldRows, () => ScalaKernels.namedFoldReuse, () => ScalaKernels.namedFoldNested,
      () => ScalaKernels.namedReuse, () => ScalaKernels.sharedExchange,
      () => ScalaKernels.sharedReuse, () => ScalaKernels.sharedFlags,
      () => ScalaKernels.phaseExchange, () => ScalaKernels.phaseReuse)
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
      val shared = actual.sharedMemory
      val bindings = locals.map(_.name) ++ indices ++ shared.map(_.name)
      assertEquals(bindings.distinct.size, bindings.size)
      shared.foreach { declaration =>
        assertNotEquals(declaration.span, SourceSpan.Unknown)
        assert(declaration.span.file.replace('\\', '/').endsWith("examples/ScalaKernels.scala"))
        assert(mapped.contains(declaration.span), s"unmapped shared declaration: $declaration")
      }
      statements.foreach { statement =>
        assertNotEquals(statement.span, SourceSpan.Unknown)
        assert(statement.span.file.replace('\\', '/').endsWith("examples/ScalaKernels.scala"))
        assert(mapped.contains(statement.span), s"unmapped statement: $statement")
      }
      val second = factory()
      assertNotEquals(locals.head.name, all(second.body).collectFirst { case d: LocalDeclaration[?] => d.local.name }.get)
      assertEquals(generated.cudaSource, CudaCodegen.generate(second).toOption.get.cudaSource)
    }
