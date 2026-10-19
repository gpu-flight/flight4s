package flight4s.ml

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule, GeneratedKernel}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.{DeviceBuffer, Kernel}

/** Explicit-DSL kernels for the MLP over one flat parameter buffer laid out as
  * w1 (row-major D x H), then b1, then w2, then b2. Accumulation orders match the CPU
  * reference exactly: biases first, then terms in index order; per-gradient sums run
  * serially over rows in one thread, so only tanh ulps separate the backends. */
private[ml] object MlpKernels:
  type PredictArgs = (DeviceBuffer[Double], DeviceBuffer[Double], DeviceBuffer[Double], Int, Int, Int)
  type TrainingForwardArgs = (DeviceBuffer[Double], DeviceBuffer[Double], DeviceBuffer[Double],
      DeviceBuffer[Double], Int, Int, Int)
  type GradientArgs = (DeviceBuffer[Double], DeviceBuffer[Double], DeviceBuffer[Double], DeviceBuffer[Double],
      DeviceBuffer[Double], DeviceBuffer[Double], Double, Double, Int, Int, Int)
  type AdamArgs = (DeviceBuffer[Double], DeviceBuffer[Double], DeviceBuffer[Double], DeviceBuffer[Double],
      Double, Double, Double, Double, Double, Double, Int)
  type SquaresArgs = (DeviceBuffer[Double], DeviceBuffer[Double], Int, Int)

  def parameterCount(inputSize: Int, hiddenSize: Int): Int =
    inputSize * hiddenSize + 2 * hiddenSize + 1

  def predictKernel: Kernel[PredictArgs] =
    kernel("flight4sMlPredict", params((input[Double]("parameters"), input[Double]("x"),
        output[Double]("predictions"), value[Int]("rows"), value[Int]("inputSize"), value[Int]("hiddenSize")))) { p =>
      val (parameters, x, predictions, rows, inputSize, hiddenSize) = p
      val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
      when(row < rows) {
        val w1Count = let("w1Count", inputSize * hiddenSize)
        val outputValue = local("outputValue", parameters(w1Count + hiddenSize + hiddenSize).read)
        gpuFor("h", literal(0), hiddenSize) { h =>
          val preactivation = local("preactivation", parameters(w1Count + h).read)
          gpuFor("d", literal(0), inputSize) { d =>
            preactivation := preactivation.read + x(row * inputSize + d).read * parameters(d * hiddenSize + h).read
          }
          outputValue := outputValue.read + tanh(preactivation.read) * parameters(w1Count + hiddenSize + h).read
        }
        predictions(row) := outputValue.read
      }
    }

  def trainingForwardKernel: Kernel[TrainingForwardArgs] =
    kernel("flight4sMlTrainingForward", params((input[Double]("parameters"), input[Double]("x"),
        output[Double]("hidden"), output[Double]("predictions"),
        value[Int]("rows"), value[Int]("inputSize"), value[Int]("hiddenSize")))) { p =>
      val (parameters, x, hidden, predictions, rows, inputSize, hiddenSize) = p
      val row = let("row", blockIdx.x * blockDim.x + threadIdx.x)
      when(row < rows) {
        val w1Count = let("w1Count", inputSize * hiddenSize)
        val outputValue = local("outputValue", parameters(w1Count + hiddenSize + hiddenSize).read)
        gpuFor("h", literal(0), hiddenSize) { h =>
          val preactivation = local("preactivation", parameters(w1Count + h).read)
          gpuFor("d", literal(0), inputSize) { d =>
            preactivation := preactivation.read + x(row * inputSize + d).read * parameters(d * hiddenSize + h).read
          }
          val activation = let("activation", tanh(preactivation.read))
          hidden(row * hiddenSize + h) := activation
          outputValue := outputValue.read + activation * parameters(w1Count + hiddenSize + h).read
        }
        predictions(row) := outputValue.read
      }
    }

  /** One thread per parameter; rowCount arrives as a Double so the output gradient is
    * computed as 2.0 * (prediction - target) / rowCount, the CPU reference's exact expression. */
  def gradientsKernel: Kernel[GradientArgs] =
    kernel("flight4sMlGradients", params((input[Double]("parameters"), input[Double]("x"), input[Double]("y"),
        input[Double]("hidden"), input[Double]("predictions"), output[Double]("gradients"),
        value[Double]("rowCount"), value[Double]("l2"),
        value[Int]("rows"), value[Int]("inputSize"), value[Int]("hiddenSize")))) { p =>
      val (parameters, x, y, hidden, predictions, gradients, rowCount, l2, rows, inputSize, hiddenSize) = p
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      val w1Count = let("w1Count", inputSize * hiddenSize)
      val total = let("total", w1Count + hiddenSize + hiddenSize + literal(1))
      when(index < total) {
        val accumulator = local("accumulator", literal(0.0))
        gpuIf(index < w1Count) {
          val d = let("d", index / hiddenSize)
          val h = let("h", index % hiddenSize)
          val outputWeight = let("outputWeight", parameters(w1Count + hiddenSize + h).read)
          gpuFor("n", literal(0), rows) { n =>
            val activation = let("activation", hidden(n * hiddenSize + h).read)
            val outputGradient = let("outputGradient",
              literal(2.0) * (predictions(n).read - y(n).read) / rowCount)
            accumulator := accumulator.read + x(n * inputSize + d).read *
              (outputGradient * outputWeight * (literal(1.0) - activation * activation))
          }
          accumulator := accumulator.read + l2 * parameters(index).read
        } {
          gpuIf(index < w1Count + hiddenSize) {
            val hBias = let("hBias", index - w1Count)
            val biasWeight = let("biasWeight", parameters(w1Count + hiddenSize + hBias).read)
            gpuFor("nBias", literal(0), rows) { n =>
              val biasActivation = let("biasActivation", hidden(n * hiddenSize + hBias).read)
              val biasGradient = let("biasGradient",
                literal(2.0) * (predictions(n).read - y(n).read) / rowCount)
              accumulator := accumulator.read +
                biasGradient * biasWeight * (literal(1.0) - biasActivation * biasActivation)
            }
          } {
            gpuIf(index < w1Count + hiddenSize + hiddenSize) {
              val hOut = let("hOut", index - w1Count - hiddenSize)
              gpuFor("nOut", literal(0), rows) { n =>
                accumulator := accumulator.read + hidden(n * hiddenSize + hOut).read *
                  (literal(2.0) * (predictions(n).read - y(n).read) / rowCount)
              }
              accumulator := accumulator.read + l2 * parameters(index).read
            } {
              gpuFor("nFinal", literal(0), rows) { n =>
                accumulator := accumulator.read + literal(2.0) * (predictions(n).read - y(n).read) / rowCount
              }
            }
          }
        }
        gradients(index) := accumulator.read
      }
    }

  def adamKernel: Kernel[AdamArgs] =
    kernel("flight4sMlAdam", params((inOut[Double]("parameters"), input[Double]("gradients"),
        inOut[Double]("moments1"), inOut[Double]("moments2"),
        value[Double]("learningRate"), value[Double]("beta1"), value[Double]("beta2"), value[Double]("epsilon"),
        value[Double]("moment1Correction"), value[Double]("moment2Correction"), value[Int]("totalParameters")))) { p =>
      val (parameters, gradients, moments1, moments2, learningRate, beta1, beta2,
        epsilon, moment1Correction, moment2Correction, totalParameters) = p
      val index = let("index", blockIdx.x * blockDim.x + threadIdx.x)
      when(index < totalParameters) {
        val gradient = let("gradient", gradients(index).read)
        val moment1 = let("moment1", beta1 * moments1(index).read + (literal(1.0) - beta1) * gradient)
        val moment2 = let("moment2", beta2 * moments2(index).read + (literal(1.0) - beta2) * gradient * gradient)
        moments1(index) := moment1
        moments2(index) := moment2
        parameters(index) := parameters(index).read -
          learningRate * (moment1 / moment1Correction) / (sqrt(moment2 / moment2Correction) + epsilon)
      }
    }

  /** Single-thread serial sum of squared weights (biases excluded), in CPU reference order. */
  def squaresKernel: Kernel[SquaresArgs] =
    kernel("flight4sMlWeightSquares", params((input[Double]("parameters"), output[Double]("result"),
        value[Int]("inputSize"), value[Int]("hiddenSize")))) { p =>
      val (parameters, result, inputSize, hiddenSize) = p
      val thread = let("thread", blockIdx.x * blockDim.x + threadIdx.x)
      when(thread === literal(0)) {
        val w1Count = let("w1Count", inputSize * hiddenSize)
        val accumulator = local("accumulator", literal(0.0))
        gpuFor("i", literal(0), w1Count) { i =>
          accumulator := accumulator.read + parameters(i).read * parameters(i).read
        }
        gpuFor("j", literal(0), hiddenSize) { j =>
          val weight = let("weight", parameters(w1Count + hiddenSize + j).read)
          accumulator := accumulator.read + weight * weight
        }
        result(literal(0)) := accumulator.read
      }
    }

  final case class GeneratedKernels(
      module: GeneratedCudaModule,
      predictDefinition: Kernel[PredictArgs],
      trainingForwardDefinition: Kernel[TrainingForwardArgs],
      gradientsDefinition: Kernel[GradientArgs],
      adamDefinition: Kernel[AdamArgs],
      squaresDefinition: Kernel[SquaresArgs],
      predict: GeneratedKernel[PredictArgs],
      trainingForward: GeneratedKernel[TrainingForwardArgs],
      gradients: GeneratedKernel[GradientArgs],
      adam: GeneratedKernel[AdamArgs],
      squares: GeneratedKernel[SquaresArgs]
  )

  def generateAll(): Either[String, GeneratedKernels] =
    val predictDefinition = predictKernel
    val trainingForwardDefinition = trainingForwardKernel
    val gradientsDefinition = gradientsKernel
    val adamDefinition = adamKernel
    val squaresDefinition = squaresKernel
    CudaCodegen.generateModule(module(kernels = Vector(predictDefinition, trainingForwardDefinition,
        gradientsDefinition, adamDefinition, squaresDefinition))) match
      case Left(error) => Left(error.message)
      case Right(generated) =>
        def named[Args <: Tuple](name: String): GeneratedKernel[Args] =
          generated.kernels.find(_.name == name)
            .getOrElse(throw IllegalStateException(s"generated module is missing kernel $name"))
            .asInstanceOf[GeneratedKernel[Args]]
        Right(GeneratedKernels(generated,
          predictDefinition, trainingForwardDefinition, gradientsDefinition, adamDefinition, squaresDefinition,
          named[PredictArgs]("flight4sMlPredict"),
          named[TrainingForwardArgs]("flight4sMlTrainingForward"),
          named[GradientArgs]("flight4sMlGradients"),
          named[AdamArgs]("flight4sMlAdam"),
          named[SquaresArgs]("flight4sMlWeightSquares")))
