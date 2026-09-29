package flight4s.frontend.examples

import scala.annotation.experimental
import flight4s.frontend.kernel
import flight4s.core.dsl.CudaDsl.{kernel as buildKernel, *}

object PrototypeKernels:
  @experimental
  @kernel
  def vectorAdd = buildKernel("annotationVectorAdd", params(
      input[Float]("left"), input[Float]("right"), output[Float]("target"), value[Int]("count"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    when(i < p._4) {
      p._3(i) := p._1(i).read + p._2(i).read
    }
  }

  @experimental
  @kernel
  def intSnapshots = buildKernel("annotationIntSnapshots", params(
      output[Int]("data"), output[Int]("saved"), output[Int]("current"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val original = p._1(i).read
    val alias = original
    p._1(i) := literal(900) + i
    p._2(i) := original + alias
    p._3(i) := p._1(i).read
  }

  @experimental
  @kernel
  def floatSnapshots = buildKernel("annotationFloatSnapshots", params(
      output[Float]("data"), output[Float]("saved"))) { p =>
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val original = p._1(i).read
    p._1(i) := literal(900.25f)
    p._2(i) := original + original
  }

  @experimental
  @kernel
  def explicitSnapshot = buildKernel("annotationExplicit", params(output[Int]("target"))) { p =>
    val saved = let("saved", threadIdx.x)
    when(threadIdx.x < literal(1)) {
      val nested = let("nested", threadIdx.x)
      p._1(threadIdx.x) := saved + nested
    }
  }
