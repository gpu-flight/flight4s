package flight4s.ml

import java.nio.file.Files
import munit.FunSuite

class MlpModelSuite extends FunSuite:
  private def randomMatrix(rows: Int, columns: Int, seed: Long): DenseMatrix =
    val random = java.util.Random(seed)
    DenseMatrix(rows, columns, Array.fill(rows * columns)(random.nextDouble() * 2.0 - 1.0))

  test("initialization is seeded Xavier uniform with zero biases and is reproducible"):
    val first = MlpModel.initialized(MlpConfig(inputSize = 5, hiddenSize = 8, seed = 99L))
    val second = MlpModel.initialized(MlpConfig(inputSize = 5, hiddenSize = 8, seed = 99L))
    val different = MlpModel.initialized(MlpConfig(inputSize = 5, hiddenSize = 8, seed = 100L))
    assertEquals(first.w1.toVector, second.w1.toVector)
    assertEquals(first.w2.toVector, second.w2.toVector)
    assert(!(first.w1.toVector == different.w1.toVector))
    assert(first.b1.forall(_ == 0.0))
    assertEquals(first.b2, 0.0)
    val w1Limit = math.sqrt(6.0 / (5 + 8))
    assert(first.w1.forall(value => value >= -w1Limit && value <= w1Limit))
    val w2Limit = math.sqrt(6.0 / (8 + 1))
    assert(first.w2.forall(value => value >= -w2Limit && value <= w2Limit))

  test("the JSON round trip preserves every field and reproduces predictions exactly"):
    val model = MlpModel.initialized(MlpConfig(inputSize = 5, hiddenSize = 8, seed = 123L))
    val restored = MlpModel.fromJson(MlpModel.toJson(model))
    assertEquals(restored.inputSize, model.inputSize)
    assertEquals(restored.hiddenSize, model.hiddenSize)
    assertEquals(restored.w1.toVector, model.w1.toVector)
    assertEquals(restored.b1.toVector, model.b1.toVector)
    assertEquals(restored.w2.toVector, model.w2.toVector)
    assertEquals(restored.b2, model.b2)
    val x = randomMatrix(rows = 9, columns = 5, seed = 7L)
    assertEquals(CpuInference.predict(restored, x).toVector, CpuInference.predict(model, x).toVector)

  test("saving to and loading from a file preserves predictions"):
    val model = MlpModel.initialized(MlpConfig(inputSize = 4, hiddenSize = 6, seed = 321L))
    val directory = Files.createTempDirectory("flight4s-ml-")
    val path = directory.resolve("model.json")
    try
      MlpModel.save(model, path)
      val restored = MlpModel.load(path)
      val x = randomMatrix(rows = 7, columns = 4, seed = 8L)
      assertEquals(CpuInference.predict(restored, x).toVector, CpuInference.predict(model, x).toVector)
    finally
      Files.deleteIfExists(path)
      Files.deleteIfExists(directory)
      ()

  test("loading validates format, version, dimensions and finiteness with clear errors"):
    val valid = MlpModel.toJson(MlpModel.initialized(MlpConfig(inputSize = 2, hiddenSize = 2, seed = 1L)))
    val wrongVersion = valid.replace("\"version\": 1", "\"version\": 2")
    assert(intercept[IllegalArgumentException](MlpModel.fromJson(wrongVersion)).getMessage.contains("version"))
    val wrongFormat = valid.replace(MlpModel.FormatName, "something-else")
    assert(intercept[IllegalArgumentException](MlpModel.fromJson(wrongFormat)).getMessage.contains("format"))
    val wrongLength = valid.replace("\"hiddenSize\": 2", "\"hiddenSize\": 3")
    assert(intercept[IllegalArgumentException](MlpModel.fromJson(wrongLength)).getMessage.contains("does not match"))
    val truncated = valid.dropRight(4)
    intercept[IllegalArgumentException](MlpModel.fromJson(truncated))
    val nonfinite = valid.replace("\"b2\": 0.0", "\"b2\": Infinity")
    intercept[IllegalArgumentException](MlpModel.fromJson(nonfinite))

  test("explicit parameter construction validates and copies its inputs"):
    val w1 = Array(0.1, 0.2)
    val model = MlpModel.fromParameters(2, 1, w1, Array(0.0), Array(0.5), 0.0)
    w1(0) = 999.0
    assertEquals(model.w1(0), 0.1)
    assert(intercept[IllegalArgumentException](
      MlpModel.fromParameters(2, 1, Array(0.1), Array(0.0), Array(0.5), 0.0)).getMessage.contains("w1 length"))
    assert(intercept[IllegalArgumentException](
      MlpModel.fromParameters(2, 1, Array(0.1, Double.NaN), Array(0.0), Array(0.5), 0.0)).getMessage.contains("nonfinite"))
