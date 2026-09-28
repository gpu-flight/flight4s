package flight4s.core.compiler

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.HexFormat

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}

final case class NvrtcCompilationKey private (hex: String):
  require(
    hex.length == 64 && hex.forall { character =>
      character >= '0' && character <= '9' ||
      character >= 'a' && character <= 'f'
    },
    "an NVRTC compilation key must be 64 lowercase hexadecimal characters"
  )

  override def toString: String = hex

object NvrtcCompilationKey:
  private val EncodingVersion = 3

  def derive(
      generated: GeneratedCudaModule,
      target: ComputeCapability,
      nvrtcVersion: NvrtcVersion,
      programName: String,
      codegenVersion: Int = CudaCodegen.ArtifactVersion
  ): NvrtcCompilationKey =
    derive(
      NvrtcCompilationInput.generated(generated, codegenVersion),
      target,
      nvrtcVersion,
      programName
    )

  def derive(
      input: NvrtcCompilationInput,
      target: ComputeCapability,
      nvrtcVersion: NvrtcVersion,
      programName: String
  ): NvrtcCompilationKey =
    require(
      input.source.nonEmpty,
      "CUDA source must not be empty"
    )
    require(programName.nonEmpty, "program name must not be empty")
    require(
      !programName.contains('\u0000'),
      "program name must not contain a null character"
    )

    val digest = MessageDigest.getInstance("SHA-256")
    val encoder = DigestEncoder(digest)
    val compilerOptions = NvrtcCompileOptions.resolve(
      input.compilerOptions,
      target
    )

    encoder.integer(EncodingVersion)
    input.provenance match
      case NvrtcSourceProvenance.DslGenerated(codegenVersion) =>
        encoder.byte(1.toByte)
        encoder.integer(codegenVersion)
      case NvrtcSourceProvenance.CallerProvidedRaw =>
        encoder.byte(2.toByte)
    encoder.integer(nvrtcVersion.major)
    encoder.integer(nvrtcVersion.minor)
    encoder.integer(target.major)
    encoder.integer(target.minor)
    encoder.string(programName)
    encoder.string(input.source)
    encoder.vector(compilerOptions.values)(encoder.string)
    encoder.vector(input.kernels) { kernel =>
      encoder.string(kernel.entryPoint)
      encoder.vector(kernel.abiDescriptors) { descriptor =>
        encoder.string(descriptor.name)
        encoder.byte(descriptor.abiType.nativeCode)
        encoder.integer(descriptor.abiType.sizeBytes)
        encoder.integer(descriptor.abiType.alignmentBytes)
      }
      encoder.optional(kernel.launchRequirements.dynamicSharedMemory) {
        requirement =>
          encoder.integer(requirement.elementSizeBytes)
          encoder.integer(requirement.elementAlignmentBytes)
      }
      encoder.optional(kernel.launchRequirements.requiredBlock) { block =>
        encoder.integer(block.x)
        encoder.integer(block.y)
        encoder.integer(block.z)
      }
    }

    NvrtcCompilationKey(HexFormat.of().formatHex(digest.digest()))

  private final class DigestEncoder(digest: MessageDigest):
    private val integerBuffer =
      ByteBuffer.allocate(java.lang.Integer.BYTES).order(ByteOrder.BIG_ENDIAN)

    def byte(value: Byte): Unit =
      digest.update(value)

    def integer(value: Int): Unit =
      integerBuffer.clear()
      integerBuffer.putInt(value)
      integerBuffer.flip()
      digest.update(integerBuffer)

    def string(value: String): Unit =
      val bytes = value.getBytes(StandardCharsets.UTF_8)
      integer(bytes.length)
      digest.update(bytes)

    def vector[A](values: Vector[A])(write: A => Unit): Unit =
      integer(values.size)
      values.foreach(write)

    def optional[A](value: Option[A])(write: A => Unit): Unit =
      value match
        case Some(present) =>
          byte(1.toByte)
          write(present)
        case None =>
          byte(0.toByte)
