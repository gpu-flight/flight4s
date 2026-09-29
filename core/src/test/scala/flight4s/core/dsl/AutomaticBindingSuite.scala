package flight4s.core.dsl

import munit.FunSuite
import scala.compiletime.testing.typeCheckErrors
import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock}

class AutomaticBindingSuite extends FunSuite:
  private given DslSourcePosition = DslSourcePosition(SourceSpan.Unknown)

  test("kernel-local bindings do not require CUDA names"):
    val errors = typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      import flight4s.core.launch.{Block as LaunchBlock}
      kernel("automatic", params(input[Float]("source"), output[Float]("out"))) { p =>
        val scratch = block.reduction[Float](LaunchBlock.x(32))
        val tile = sharedArray2D[Float](rows = 2, columns = 32, rowStride = 33)
        val cube = sharedArray3D[Float](depth = 2, rows = 2, columns = 32)
        val temporary = localArray[Float](2)
        val index = let { blockIdx.x * blockDim.x + threadIdx.x }
        val before = let(p._1(index).read)
        val counter = local(literal(0))
        gpuFor(literal(0), literal(2)) { iteration => counter := iteration }
        gpuRange(literal(0), literal(2)).foreach { i => temporary(i) := before }
        val sum = scratch.sum(before)
        val difference = scratch.reduceTree(before)(_ - _)
        p._2(index) := sum + difference
      }
    """)
    assertEquals(errors.map(_.message), Nil)

  test("unnamed let snapshots a load once and retains ordinary val expression semantics"):
    val definition = kernel("snapshot", params(inOut[Int]("data"), output[Int]("out"))) { p =>
      val snapshot = let(p._1(literal(0)).read)
      val expression = p._1(literal(0)).read
      p._1(literal(0)) := literal(99)
      p._2(literal(0)) := snapshot + snapshot
      p._2(literal(1)) := expression
    }
    val source = CudaCodegen.generate(definition).toOption.get.cudaSource
    assertEquals(source, """extern "C" __global__ void snapshot(int* data, int* out) {
  int flight4s_auto_value_0 = data[0];
  data[0] = 99;
  out[0] = (flight4s_auto_value_0 + flight4s_auto_value_0);
  out[1] = data[0];
}
""")

  test("emitted names avoid parameters constants and later explicit declarations"):
    val table = constantArray[Int]("flight4s_auto_value_1", 1)
    val definition = kernel("collisions", params(value[Int]("flight4s_auto_value_0"), output[Int]("out"))) { p =>
      val automatic = let(threadIdx.x)
      val explicit = local("flight4s_auto_value_2", p._1)
      val array = localArray[Int]("flight4s_auto_value_4", 1)
      val shared = sharedArray[Int]("flight4s_auto_value_5", 1)
      gpuFor("flight4s_auto_value_3", literal(0), p._1) { i =>
        array(literal(0)) := automatic + explicit.read + table(literal(0)).read
        shared(literal(0)) := array(literal(0)).read
        p._2(i) := shared(literal(0)).read
      }
    }
    val source = CudaCodegen.generateModule(module(Vector(table), Vector(definition))).toOption.get.cudaSource
    assert(source.contains("int flight4s_auto_value_6 = static_cast<int>(threadIdx.x);"), source)
    assert(!source.contains("$flight4s$"))

  test("automatic bindings preserve source positions"):
    val span = SourceSpan("Automatic.scala", 10, 2, 10, 30)
    val definition = kernel("position") {
      given DslSourcePosition = DslSourcePosition(span)
      val stored = let(threadIdx.x)
      local(stored)
      ()
    }
    val declarations = definition.body.statements.map(_.asInstanceOf[LocalDeclaration[?]])
    assert(declarations.forall(d => d.span == span && d.local.span == span))
    assertEquals(declarations(1).initial.span, span)
    assert(CudaCodegen.generate(definition).toOption.get.sourceMap.entries.exists(_.sourceSpan == span))

  test("separate and concurrent constructions emit identical artifacts despite distinct internal owners"):
    val signature = params(output[Int]("out"))
    def build() =
      val definition = kernel("deterministic", signature) { p =>
        val scratch = block.reduction[Int](LaunchBlock.x(2))
        val rank = let(threadIdx.x)
        val result = scratch.sum(rank)
        p._1(rank) := result
      }
      CudaCodegen.generate(definition).toOption.get
    val expected = build()
    assertEquals(build(), expected)
    val jobs = Vector.fill(16)(java.util.concurrent.CompletableFuture.supplyAsync(() => build()))
    jobs.foreach(job => assertEquals(job.join(), expected))

  test("automatic locals escaping a nested scope or another kernel are not rebound"):
    var escaped = Option.empty[Expr[Int]]
    val nested = kernel("nested", params(output[Int]("out"))) { p =>
      scoped { escaped = Some(let(threadIdx.x)) }
      let(threadIdx.y)
      p._1(literal(0)) := escaped.get
    }
    assertEquals(KernelValidator.validate(nested).errors.map(_.code), Vector(ValidationCode.UnboundLocal))
    kernel("owner") { escaped = Some(let(threadIdx.x)) }
    val foreign = kernel("foreign", params(output[Int]("out"))) { p =>
      val own = let(threadIdx.y)
      p._1(literal(0)) := own + escaped.get
    }
    assertEquals(KernelValidator.validate(foreign).errors.map(_.code), Vector(ValidationCode.UnboundLocal))

  test("block reductions allocate one scratch array and independent results with unchanged launch requirements"):
    val definition = kernel("reuse", params(output[Int]("out"))) { p =>
      val reduction = block.reduction[Int](LaunchBlock.x(32))
      val first = reduction.sum(threadIdx.x)
      val second = reduction.reduceTree(threadIdx.x)(_ - _)
      p._1(threadIdx.x) := first + second
    }
    assertEquals(definition.ir.sharedMemory.size, 1)
    assertEquals(definition.ir.requiredBlock, Some(LaunchBlock.x(32)))
    assert(KernelValidator.validate(definition).isValid)
    val locals = definition.body.statements.collect { case d: LocalDeclaration[?] => d.local.name }
    assertEquals(locals.distinct.size, 2)
    val source = CudaCodegen.generate(definition).toOption.get.cudaSource
    assertEquals("__syncthreads\\(\\);".r.findAllIn(source).size, 14)
    assert(!source.contains("$flight4s$"))

  test("automatic scratch from another kernel cannot alias a same-position scratch declaration"):
    var captured = Option.empty[BlockReduction[Int]]
    kernel("owner") { captured = Some(block.reduction[Int](LaunchBlock.x(32))) }
    val other = kernel("other", params(output[Int]("out"))) { p =>
      val own = block.reduction[Int](LaunchBlock.x(32))
      val first = own.sum(threadIdx.x)
      val second = captured.get.sum(threadIdx.x)
      p._1(threadIdx.x) := first + second
    }
    assert(KernelValidator.validate(other).errors.exists(_.code == ValidationCode.UnknownSharedMemory))

  test("internal binding keys are not accepted as public ABI names"):
    var key = ""
    kernel("owner") { key = local(literal(0)).name }
    assert(!KernelValidator.validate(kernel(key) {}).isValid)
    assert(!KernelValidator.validate(kernel("parameter", params(value[Int](key))) { _ => () }).isValid)
    assert(!ModuleValidator.validate(module(Vector(constantArray[Int](key, 1)))).isValid)

  test("malformed internal-looking names fail validation instead of crashing emission"):
    for name <- Vector("$flight4s$1$99999999999999999999999$value", "$flight4s$-1$0$value", "$ordinary") do
      val definition = kernel("invalid") { local(name, literal(0)); () }
      assert(!KernelValidator.validate(definition).isValid)
      assert(CudaCodegen.generate(definition).isLeft)

  test("unnamed static and dynamic arrays retain rank layout and launch metadata"):
    val definition = kernel("arrays", params(output[Int]("out"))) { p =>
      val one = sharedArray[Int](32)
      val two = sharedArray2D[Int](2, 32, 33)
      val three = sharedArray3D[Int](2, 2, 32, 33)
      val dynamic = dynamicSharedArray[Int]()
      val local = localArray[Int](2)
      one(threadIdx.x) := threadIdx.x
      two(literal(0), threadIdx.x) := one(threadIdx.x).read
      three(literal(0), literal(0), threadIdx.x) := two(literal(0), threadIdx.x).read
      dynamic(threadIdx.x) := three(literal(0), literal(0), threadIdx.x).read
      local(literal(0)) := dynamic(threadIdx.x).read
      p._1(threadIdx.x) := local(literal(0)).read
    }
    val generated = CudaCodegen.generate(definition).toOption.get
    assert(generated.cudaSource.contains("[2][33]"))
    assert(generated.cudaSource.contains("[2][2][33]"))
    assert(generated.launchRequirements.dynamicSharedMemory.nonEmpty)
    assert(!generated.cudaSource.contains("$flight4s$"))

  test("unnamed loops and expression reductions share the normal validation and lowering path"):
    val definition = kernel("ranges", params(output[Int]("out"))) { p =>
      val total = local(literal(0))
      gpuFor(literal(0), literal(4), 2) { outer =>
        gpuFor(literal(0), literal(3)) { inner => accumulate(total, outer + inner) }
      }
      val expression = gpuRange(literal(0), literal(5)).by(2).map(identity).sum(literal(0))
      p._1(literal(0)) := total.read + expression
    }
    val source = CudaCodegen.generate(definition).toOption.get.cudaSource
    assert(source.contains(" += 2LL"))
    assert(source.contains(" += "))
    assert(!source.contains("$flight4s$"))

  test("expression callbacks and nested shared declarations still reject statements"):
    val expressionError = intercept[DslError] {
      kernel("expression") {
        gpuRange(literal(0), literal(2)).map(i => let(i)).sum(literal(0))
        ()
      }
    }
    assertEquals(expressionError.code, DslErrorCode.StatementInsideExpression)
    val sharedError = intercept[DslError] {
      kernel("nestedShared") { scoped { sharedArray[Int](32); () } }
    }
    assertEquals(sharedError.code, DslErrorCode.SharedMemoryDeclarationOutsideKernelBody)

  test("unnamed let still requires a builder and returns a read-only expression"):
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      let(literal(1))
    """).nonEmpty)
    assert(typeCheckErrors("""
      import flight4s.core.dsl.CudaDsl.*
      kernel("write") { val x = let(literal(1)); x := literal(2) }
    """).nonEmpty)
