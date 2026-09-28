package flight4s.runtime.cuda

import munit.FunSuite
import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule, GeneratedKernel}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}
import flight4s.core.types.*

class CudaScopedAtomicJniSuite extends FunSuite:
  private val orders = MemoryOrder.values.toVector

  test("all scoped atomic scalar types orders and storage variants execute without headers"):
    withContext { context =>
      for storage <- Vector("device", "blockGlobal", "shared") do
        scalar(context, storage, (i: Int) => i)
        scalar(context, storage, UInt.fromBits)
        scalar(context, storage, (i: Int) => i.toFloat)
        scalar(context, storage, (i: Int) => i.toDouble)
    }

  test("integer min max bitwise and compare-exchange return old values on success and failure"):
    withContext { context =>
      for storage <- Vector("device", "blockGlobal", "shared") do
        integral(context, storage, (i: Int) => i)
        integral(context, storage, UInt.fromBits)
    }

  test("contended device and shared block counters allocate unique tickets and stable snapshots"):
    withContext { context =>
      for order <- orders do
        val definition = kernel("tickets", params(output[Int]("counter"), output[Int]("out"))) { p =>
          val shared = sharedArray[Int]("sharedCounter", 1)
          val ref = atomic.block(shared(literal(0)))
          when(threadIdx.x === literal(0)) { ref.store(literal(0), MemoryOrder.Relaxed) }
          barrier()
          val i = let("i", blockIdx.x * blockDim.x + threadIdx.x)
          val globalTicket = atomic.device(p._1(literal(0))).fetchAdd("globalTicket", literal(1), order)
          val blockTicket = ref.fetchAdd("blockTicket", literal(1), order)
          barrier()
          val count = ref.load("count", MemoryOrder.Acquire)
          p._2(i) := globalTicket
          p._2(literal(512) + i) := blockTicket
          p._2(literal(1024) + i) := count
          p._2(literal(1536) + i) := globalTicket
        }
        val generated = CudaCodegen.generate(definition).toOption.get
        val module = compile(context, generated)
        val counter = context.allocate[Int](1).toOption.get
        val out = context.allocate[Int](2056).toOption.get
        val stream = context.createStream().toOption.get
        try
          val function = module.function(generated).toOption.get
          for explicit <- Vector(false, true) do
            assertEquals(counter.copyFrom(Array(0)), Right(()))
            assertEquals(out.copyFrom(Array.fill(2056)(-1)), Right(()))
            val args = definition.bind((counter, out))
            val config = LaunchConfig(Grid.x(8), LaunchBlock.x(64))
            assertEquals(if explicit then function.launch(args, config, stream) else function.launch(args, config), Right(()))
            assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
            val actual = out.copyToArray().toOption.get
            assertEquals(counter.copyToArray().toOption.get.toVector, Vector(512))
            assertEquals(actual.take(512).sorted.toVector, (0 until 512).toVector)
            for block <- 0 until 8 do
              assertEquals(actual.slice(512 + block * 64, 512 + (block + 1) * 64).sorted.toVector, (0 until 64).toVector)
            assert(actual.slice(1024, 1536).forall(_ == 64))
            assertEquals(actual.slice(1536, 2048).toVector, actual.take(512).toVector)
            assert(actual.drop(2048).forall(_ == -1))
        finally
          counter.close(); out.close(); stream.close(); module.close()
    }

  test("release publication and acquire observation never expose an unpublished payload"):
    withContext { context =>
      for deviceScope <- Vector(false, true) do
        val definition = kernel("publish", params(output[Int]("payload"), output[Int]("flags"), output[Int]("out"))) { p =>
          val producer = blockIdx.x
          val flag = if deviceScope then atomic.device(p._2(producer)) else atomic.block(p._2(producer))
          when(threadIdx.x === literal(0)) {
            p._1(producer) := literal(1000) + producer
            flag.store(literal(1), MemoryOrder.Release)
          }
          val watched = if deviceScope then (blockIdx.x + literal(1)) % literal(16) else blockIdx.x
          val observedFlag = if deviceScope then atomic.device(p._2(watched)) else atomic.block(p._2(watched))
          val i = let("i", blockIdx.x * blockDim.x + threadIdx.x)
          gpuFor("poll", literal(0), literal(128)) { _ =>
            val ready = observedFlag.load("ready", MemoryOrder.Acquire)
            when(ready === literal(1)) { p._3(i) := p._1(watched).read }
          }
        }
        val generated = CudaCodegen.generate(definition).toOption.get
        val module = compile(context, generated)
        val payload = context.allocate[Int](16).toOption.get
        val flags = context.allocate[Int](16).toOption.get
        val out = context.allocate[Int](1024).toOption.get
        val stream = context.createStream().toOption.get
        try
          val function = module.function(generated).toOption.get
          for explicit <- Vector(false, true) do
            assertEquals(payload.copyFrom(Array.fill(16)(-99)), Right(()))
            assertEquals(flags.copyFrom(Array.fill(16)(0)), Right(()))
            assertEquals(out.copyFrom(Array.fill(1024)(-1)), Right(()))
            val args = definition.bind((payload, flags, out))
            val config = LaunchConfig(Grid.x(16), LaunchBlock.x(64))
            assertEquals(if explicit then function.launch(args, config, stream) else function.launch(args, config), Right(()))
            assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
            val actual = out.copyToArray().toOption.get
            assert(actual.exists(_ >= 1000), "no acquire observed publication")
            for i <- actual.indices do
              val watched = if deviceScope then (i / 64 + 1) % 16 else i / 64
              assert(actual(i) == -1 || actual(i) == 1000 + watched, s"index=$i value=${actual(i)}")
            assert(flags.copyToArray().toOption.get.forall(_ == 1))
        finally
          payload.close(); flags.close(); out.close(); stream.close(); module.close()
    }

  private def scalar[T](context: CudaContext, storage: String, value: Int => T)(using
      atomicType: AtomicValueType[T], codec: CudaHostCodec[T], tag: scala.reflect.ClassTag[T]
  ): Unit =
    val definition = kernel("scalarAtomics", params(output[T]("counter"), output[T]("out"))) { p =>
      val shared = sharedArray[T]("scratch", 1)
      val ref = storage match
        case "device" => atomic.device(p._1(literal(0)))
        case "blockGlobal" => atomic.block(p._1(literal(0)))
        case _ => atomic.block(shared(literal(0)))
      for (order, index) <- orders.zipWithIndex do scoped {
        ref.store(literal(value(10)), MemoryOrder.Relaxed)
        val loaded = ref.load("loaded", if order.validForLoad then order else MemoryOrder.Relaxed)
        val exchanged = ref.exchange("exchanged", literal(value(20)), order)
        val added = ref.fetchAdd("added", literal(value(3)), order)
        val subtracted = ref.fetchSub("subtracted", literal(value(4)), order)
        val result = ref.load("result", MemoryOrder.Relaxed)
        ref.store(literal(value(30)), if order.validForStore then order else MemoryOrder.Relaxed)
        val stored = ref.load("stored", MemoryOrder.SequentiallyConsistent)
        for (expr, column) <- Vector(loaded, exchanged, added, subtracted, result, stored, loaded).zipWithIndex do
          p._2(literal(index * 7 + column)) := expr
      }
    }
    verify(context, definition, Vector.fill(5)(Vector(10, 10, 20, 23, 19, 30, 10)).flatten.map(value), value(-1))

  private def integral[T](context: CudaContext, storage: String, value: Int => T)(using
      atomicType: AtomicIntegralType[T], codec: CudaHostCodec[T], tag: scala.reflect.ClassTag[T]
  ): Unit =
    val definition = kernel("integerAtomics", params(output[T]("counter"), output[T]("out"))) { p =>
      val shared = sharedArray[T]("scratch", 1)
      val ref = storage match
        case "device" => atomic.device(p._1(literal(0)))
        case "blockGlobal" => atomic.block(p._1(literal(0)))
        case _ => atomic.block(shared(literal(0)))
      for (order, index) <- orders.zipWithIndex do scoped {
        ref.store(literal(value(12)), MemoryOrder.Relaxed)
        val min = ref.fetchMin("minimum", literal(value(5)), order)
        val max = ref.fetchMax("maximum", literal(value(9)), order)
        val and = ref.fetchAnd("oldAnd", literal(value(6)), order)
        val or = ref.fetchOr("oldOr", literal(value(10)), order)
        val xor = ref.fetchXor("oldXor", literal(value(3)), order)
        val failureOrder = if order == MemoryOrder.SequentiallyConsistent then order
          else if order.permitsFailure(MemoryOrder.Acquire) then MemoryOrder.Acquire else MemoryOrder.Relaxed
        val success = ref.compareExchange("success", literal(value(9)), literal(value(4)), order, failureOrder)
        val failure = ref.compareExchange("failure", literal(value(9)), literal(value(77)), order, failureOrder)
        val result = ref.load("result", MemoryOrder.Relaxed)
        for (expr, column) <- Vector(min, max, and, or, xor, success, failure, result, min).zipWithIndex do
          p._2(literal(index * 9 + column)) := expr
      }
    }
    verify(context, definition, Vector.fill(5)(Vector(12, 5, 9, 0, 10, 9, 4, 4, 12)).flatten.map(value), value(-1))

  private def verify[T](context: CudaContext,
      definition: Kernel[(flight4s.core.ir.DeviceBuffer[T], flight4s.core.ir.DeviceBuffer[T])],
      expected: Vector[T], sentinel: T)(using CudaType[T], CudaHostCodec[T], scala.reflect.ClassTag[T]): Unit =
    val generated = CudaCodegen.generate(definition).toOption.get
    val module = compile(context, generated)
    val counter = context.allocate[T](1).toOption.get
    val out = context.allocate[T](expected.size + 8).toOption.get
    val stream = context.createStream().toOption.get
    try
      val function = module.function(generated).toOption.get
      for explicit <- Vector(false, true) do
        assertEquals(out.copyFrom(Array.fill(expected.size + 8)(sentinel)), Right(()))
        val args = definition.bind((counter, out))
        val config = LaunchConfig(Grid.x(1), LaunchBlock.x(1))
        assertEquals(if explicit then function.launch(args, config, stream) else function.launch(args, config), Right(()))
        assertEquals(if explicit then stream.synchronize() else context.synchronize(), Right(()))
        assertEquals(out.copyToArray().toOption.get.toVector, expected ++ Vector.fill(8)(sentinel))
    finally
      counter.close(); out.close(); stream.close(); module.close()

  private def compile(context: CudaContext, generated: GeneratedKernel[?]): CudaModule =
    val artifact = NvrtcCompiler.compile(GeneratedCudaModule(generated.cudaSource, generated.sourceMap,
      generated.compilerOptions, Vector(generated)), context.computeCapability, "scoped_atomics.cu")
      .fold(f => fail(f.message + "\n" + f.compileLog), identity)
    context.load(artifact).fold(f => fail(f.message), identity)

  private def withContext(body: CudaContext => Unit): Unit =
    assume(sys.props.contains("flight4s.cuda.native.path"), "set flight4s.cuda.native.path to run JNI tests")
    val context = CudaContext.open(0).fold(f => fail(f.message), identity)
    try body(context)
    finally context.close()
