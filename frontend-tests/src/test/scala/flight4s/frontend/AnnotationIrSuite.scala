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
