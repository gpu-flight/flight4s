package flight4s.ml

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{DeviceBuffer, Kernel}

/** Explicit-DSL kernels for the MLP; accumulation order matches the CPU reference exactly:
  * each pre-activation starts from its bias and adds terms in feature-index order, and the
  * output accumulates hidden terms in hidden-index order. */
private[ml] object MlpKernels:
  type ForwardArgs = (DeviceBuffer[Double], DeviceBuffer[Double], DeviceBuffer[Double], DeviceBuffer[Double],
      Double, DeviceBuffer[Double], Int, Int, Int)

  def forward: Kernel[ForwardArgs] =
    kernel("flight4sMlForward", params((input[Double]("x"), input[Double]("w1"), input[Double]("b1"),
        input[Double]("w2"), value[Double]("b2"), output[Double]("predictions"),
        value[Int]("rows"), value[Int]("inputSize"), value[Int]("hiddenSize")))) { p =>
      val (x, w1, b1, w2, b2, predictions, rows, inputSize, hiddenSize) = p
      val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
      when(row < rows) {
        val outputValue = local("outputValue", b2)
        gpuFor("h", literal(0), hiddenSize) { h =>
          val preactivation = local("preactivation", b1(h).read)
          gpuFor("d", literal(0), inputSize) { d =>
            preactivation := preactivation.read + x(row * inputSize + d).read * w1(d * hiddenSize + h).read
          }
          outputValue := outputValue.read + tanh(preactivation.read) * w2(h).read
        }
        predictions(row) := outputValue.read
      }
    }
