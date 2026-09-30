package flight4s.frontend

import munit.FunSuite
import flight4s.core.ir.*

class TupleSignatureCompilerSuite extends FunSuite:
  for (name, call) <- Vector(
      "core inferred tuple" -> "flight4s.core.dsl.CudaDsl.params((value[Int](\"a\"), value[Float](\"b\")))",
      "core explicit cons" -> "flight4s.core.dsl.CudaDsl.params(value[Int](\"a\") *: value[Float](\"b\") *: EmptyTuple)",
      "direct tuple alias" -> "flight4s.core.dsl.CudaDsl.paramsTuple((value[Int](\"a\"), value[Float](\"b\")))",
      "signature factory" -> "flight4s.core.ir.KernelSignature.fromTuple((value[Int](\"a\"), value[Float](\"b\")))"
  ) do
    test(s"$name passes both compiler checks"):
      CompilerHarness.withDirectory { directory =>
        val result = CompilerHarness.compile(directory, "CoreTuple", s"""
          package tuplefixture
          import flight4s.core.dsl.CudaDsl.*
          import flight4s.core.ir.*
          class Definitions:
            val signature = $call
            val typed: KernelSignature[(Int, Float)] = signature
        """)
        assertEquals(result.errors, Vector.empty)
      }

  for (name, builder) <- Vector(
      "independent argument inference" -> "def accept[Args <: Tuple, Params <: Tuple](signature: KernelSignature[Args] { type Bindings = Params })(body: DeviceBindingsOf[Params] => Unit): Unit = ()",
      "derived argument inference" -> "def accept[Params <: Tuple](signature: KernelSignature[KernelArgumentsOf[Params]] { type Bindings = Params })(body: DeviceBindingsOf[Params] => Unit): Unit = ()"
  ) do
    test(s"$name passes both compiler checks"):
      CompilerHarness.withDirectory { directory =>
        val result = CompilerHarness.compile(directory, "Inference", s"""
          package tuplefixture
          import flight4s.core.ir.*
          import flight4s.frontend.ScalaKernel.*
          class Definitions:
            $builder
            def definition = accept(params((value[Int]("a"), value[Float]("b")))) { p =>
              val a: Int = p._1
              val b: Float = p._2
            }
        """)
        assertEquals(result.errors, Vector.empty)
      }

  for call <- Vector("params", "flight4s.core.dsl.CudaDsl.paramsTuple", "flight4s.core.ir.KernelSignature.fromTuple") do
    test(s"quoted seven parameter tuple via $call passes both compiler checks"):
      CompilerHarness.withDirectory { directory =>
        val result = CompilerHarness.compile(directory, "TupleKernel", s"""
          package tuplefixture
          import scala.annotation.experimental
          import flight4s.frontend.ScalaKernel.*
          class Definitions:
            @experimental
            def definition = kernel("tupleKernel", $call((
              input[Float]("data"), input[Int]("lengths"), output[Float]("target"),
              value[Int]("rows"), value[Int]("columns"),
              value[Float]("threshold"), value[Float]("seed")))) { p =>
              val row = blockIdx.x * blockDim.x + threadIdx.x
              if row < p._4 then
                val values = for column <- deviceRange(0, p._5)
                  if column < p._2(row) if p._1(row * p._5 + column) > p._6
                yield p._1(row * p._5 + column)
                val total = values.foldLeft(p._7)((sum, item) => sum + item)
                p._3(row) = total
            }
        """)
        assertEquals(result.errors, Vector.empty)
        CompilerHarness.withClasses(Seq(result.classes)) { loader =>
          val definition = loader.loadClass("tuplefixture.Definitions")
          val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
            .asInstanceOf[Kernel[Tuple]]
          assert(KernelValidator.validate(staged).isValid)
          assertEquals(staged.params.map(_.name), Vector("data", "lengths", "target", "rows", "columns", "threshold", "seed"))
        }
        val caller = CompilerHarness.compile(directory, "TupleCaller", """
          package tuplefixture
          import scala.annotation.experimental
          import flight4s.core.ir.*
          class Caller:
            @experimental
            def definition: Kernel[(DeviceBuffer[Float], DeviceBuffer[Int], DeviceBuffer[Float], Int, Int, Float, Float)] =
              new Definitions().definition
        """, dependencies = Seq(result.classes))
        assertEquals(caller.errors, Vector.empty)
      }

  for arity <- Vector(0, 1, 2, 6, 7, 23) do
    test(s"quoted tuple arity $arity retains bindings and separately compiled launch types"):
      val descriptors = (1 to arity).map(i => s"value[Int](\"p$i\")")
      val tuple = arity match
        case 0 => "EmptyTuple"
        case 1 => s"Tuple1(${descriptors.head})"
        case _ => descriptors.mkString("(", ", ", ")")
      val body = if arity == 0 then "val i = threadIdx.x"
        else if arity <= 22 then s"val i = p._1 + p._$arity"
        else s"val i = p(0) + p(${arity - 1})"
      val argsType = List.fill(arity)("Int").mkString("", " *: ", if arity == 0 then "EmptyTuple" else " *: EmptyTuple")
      CompilerHarness.withDirectory { directory =>
        val result = CompilerHarness.compile(directory, "ArityKernel", s"""
          package tuplefixture
          import scala.annotation.experimental
          import flight4s.frontend.ScalaKernel.*
          class Definitions:
            @experimental
            def definition = kernel("arityKernel", params($tuple)) { p => $body }
        """)
        assertEquals(result.errors, Vector.empty)
        CompilerHarness.withClasses(Seq(result.classes)) { loader =>
          val definition = loader.loadClass("tuplefixture.Definitions")
          val staged = definition.getMethod("definition").invoke(definition.getConstructor().newInstance())
            .asInstanceOf[Kernel[Tuple]]
          assert(KernelValidator.validate(staged).isValid)
          assertEquals(staged.params.map(_.name), (1 to arity).map(i => s"p$i").toVector)
        }
        val caller = CompilerHarness.compile(directory, "ArityCaller", s"""
          package tuplefixture
          import scala.annotation.experimental
          import flight4s.core.ir.*
          class Caller:
            @experimental
            def definition: Kernel[$argsType] = new Definitions().definition
        """, dependencies = Seq(result.classes))
        assertEquals(caller.errors, Vector.empty)
      }

  test("generic helpers and stored signatures retain supplied evidence and exact launch types"):
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "GenericSignature", """
        package tuplefixture
        import flight4s.core.ir.*
        import flight4s.core.dsl.CudaDsl.*
        object Signatures:
          def generic[P <: Tuple](bindings: P)(using KernelParamTuple[P]) = paramsTuple(bindings)
          def explicit[P <: Tuple](bindings: P)(using tuple: KernelParamTuple[P]) =
            KernelSignature.fromTupleWithEvidence(bindings)(using tuple)
          val inferred = generic((input[Float]("data"), output[Float]("target"), value[Int]("count")))
          val supplied = explicit(inferred.bindings)
      """)
      assertEquals(result.errors, Vector.empty)
      val caller = CompilerHarness.compile(directory, "GenericCaller", """
        package tuplefixture
        import scala.annotation.experimental
        import flight4s.core.ir.*
        import flight4s.frontend.ScalaKernel.*
        class Caller:
          @experimental
          def definition: Kernel[(DeviceBuffer[Float], DeviceBuffer[Float], Int)] =
            kernel("genericKernel", Signatures.supplied) { (data, target, count) =>
              val i = threadIdx.x
              if i < count then target(i) = data(i)
            }
      """, dependencies = Seq(result.classes))
      assertEquals(caller.errors, Vector.empty)
    }

  test("tuple signatures still reject wrong launch arity order buffer types and read only stores"):
    val source = """
      package tuplefixture
      import scala.annotation.experimental
      import flight4s.core.ir.*
      import flight4s.frontend.ScalaKernel.*
      class Definitions:
        @experimental
        def definition = kernel("copy", params((input[Float]("data"), output[Float]("target"), value[Int]("count")))) { p =>
          if threadIdx.x < p._3 then p._2(threadIdx.x) = p._1(threadIdx.x)
        }
    """
    CompilerHarness.withDirectory { directory =>
      val result = CompilerHarness.compile(directory, "ValidSignature", source)
      assertEquals(result.errors, Vector.empty)
      for (name, args) <- Vector("Arity" -> "(data, target)", "Order" -> "(1, data, target)",
          "Buffer" -> "(wrong, target, 1)", "Scalar" -> "(data, target, 1.0f)") do
        val rejected = CompilerHarness.compile(directory, name, s"""
          package tuplefixture
          import scala.annotation.experimental
          import flight4s.core.ir.*
          class $name:
            @experimental
            def launch(data: DeviceBuffer[Float], target: DeviceBuffer[Float], wrong: DeviceBuffer[Double]) =
              new Definitions().definition.bind($args)
        """, dependencies = Seq(result.classes))
        assert(rejected.errors.exists(_.contains("Found:")), rejected.errors.mkString("\n"))
      val readOnly = CompilerHarness.compile(directory, "ReadOnly", source.replace("p._2(threadIdx.x) =", "p._1(threadIdx.x) ="))
      assert(readOnly.errors.exists(_.contains("ReadWrite")), readOnly.errors.mkString("\n"))
      val invalid = CompilerHarness.compile(directory, "Invalid", """
        import flight4s.core.dsl.CudaDsl.*
        val signature = params((value[Int]("count"), "not a kernel parameter"))
      """)
      assert(invalid.errors.exists(_.contains("KernelParam")), invalid.errors.mkString("\n"))
    }
