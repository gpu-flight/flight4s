package flight4s.runtime.cuda

import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.attribute.BasicFileAttributes
import java.nio.file.{
  AtomicMoveNotSupportedException,
  FileVisitResult,
  Files,
  Path,
  SimpleFileVisitor,
  StandardCopyOption
}
import java.security.MessageDigest
import java.util.{Base64, HexFormat}

import flight4s.core.codegen.GeneratedCudaModule
import flight4s.core.compiler.*

/** A failure encountered while reading or writing a persistent NVRTC artifact. */
sealed trait NvrtcArtifactStoreError:
  def message: String

final case class NvrtcArtifactStoreIoFailure(
    operation: String,
    path: Path,
    reason: String
) extends NvrtcArtifactStoreError:
  override def message: String =
    s"NVRTC artifact store could not $operation at $path: $reason"

final case class NvrtcArtifactStoreInvalidEntry(
    key: NvrtcCompilationKey,
    path: Path,
    reason: String
) extends NvrtcArtifactStoreError:
  override def message: String =
    s"NVRTC artifact store entry $key at $path is invalid: $reason"

final case class NvrtcArtifactStoreUnsupportedSchema(
    key: NvrtcCompilationKey,
    path: Path,
    foundVersion: Int
) extends NvrtcArtifactStoreError:
  override def message: String =
    s"NVRTC artifact store entry $key at $path uses unsupported schema " +
      s"version $foundVersion"

final case class NvrtcArtifactStoreSourceMismatch(
    key: NvrtcCompilationKey,
    path: Path
) extends NvrtcArtifactStoreError:
  override def message: String =
    s"NVRTC artifact store entry $key at $path does not match the supplied CUDA source"

/**
 * Persistent storage for successful NVRTC compilation artifacts.
 *
 * Entries are atomically published under a deterministic compilation key. The
 * stored payload deliberately excludes source-map metadata: callers supply the
 * current [[NvrtcCompilationInput]] when loading, so typed metadata and
 * diagnostics retain the current caller's input.
 */
final class NvrtcArtifactStore private[cuda] (root: Path):
  require(root != null, "NVRTC artifact store root must not be null")

  import NvrtcArtifactStore.*

  /** Normalized root directory used for all artifact entries. */
  val directory: Path = root.toAbsolutePath.normalize

  /**
   * Loads an artifact and rebinds it to `generated`.
   *
   * A missing entry is represented by `Right(None)`. Corrupt, incompatible,
   * and unreadable entries remain distinguishable so a future read-through
   * cache can turn them into cache misses with its own eviction policy.
   */
  def load(
      key: NvrtcCompilationKey,
      generated: GeneratedCudaModule
  ): Either[NvrtcArtifactStoreError, Option[NvrtcArtifact]] =
    load(key, NvrtcCompilationInput.generated(generated))

  /** Loads generated or raw PTX and rebinds it to the current typed input. */
  def load(
      key: NvrtcCompilationKey,
      input: NvrtcCompilationInput
  ): Either[NvrtcArtifactStoreError, Option[NvrtcArtifact]] =
    val entry = entryPath(key)
    try
      if !Files.exists(entry) then Right(None)
      else loadEntry(key, input, entry).map(Some(_))
    catch
      case exception: IOException =>
        Left(ioFailure("read artifact", entry, exception))
      case exception: SecurityException =>
        Left(ioFailure("read artifact", entry, exception))

  /**
   * Writes one successful compilation artifact if no entry for `key` exists.
   *
   * A same-key entry is immutable. Concurrent writers may race to publish an
   * equivalent entry; the loser discards its temporary directory and succeeds.
   */
  def store(
      key: NvrtcCompilationKey,
      artifact: NvrtcArtifact
  ): Either[NvrtcArtifactStoreError, Unit] =
    val entry = entryPath(key)
    validateIdentity(
      key,
      entry,
      artifact.input,
      artifact.nvrtcVersion,
      artifact.target,
      artifact.compilerOptions,
      artifact.programName
    ).flatMap(_ => writeArtifact(key, artifact, entry))

  /** Removes the completed entry for `key`, if one exists. */
  def remove(key: NvrtcCompilationKey): Either[NvrtcArtifactStoreError, Unit] =
    val entry = entryPath(key)
    try
      deleteRecursively(entry)
      Right(())
    catch
      case exception: IOException =>
        Left(ioFailure("remove artifact", entry, exception))
      case exception: SecurityException =>
        Left(ioFailure("remove artifact", entry, exception))

  /**
   * Removes every completed entry while retaining the configured store root.
   *
   * This method does not coordinate with writers in other processes.
   */
  def clear(): Either[NvrtcArtifactStoreError, Unit] =
    try
      if Files.exists(directory) then
        val entries = Files.list(directory)
        try
          val iterator = entries.iterator()
          while iterator.hasNext do deleteRecursively(iterator.next())
        finally entries.close()
      Right(())
    catch
      case exception: IOException =>
        Left(ioFailure("clear artifacts", directory, exception))
      case exception: SecurityException =>
        Left(ioFailure("clear artifacts", directory, exception))

  private[cuda] def entryPath(key: NvrtcCompilationKey): Path =
    val keyText = key.toString
    directory.resolve(keyText.take(2)).resolve(keyText)

  private def loadEntry(
      key: NvrtcCompilationKey,
      input: NvrtcCompilationInput,
      entry: Path
  ): Either[NvrtcArtifactStoreError, NvrtcArtifact] =
    val manifestPath = entry.resolve(ManifestFileName)
    for
      manifestBytes <- readFile(key, manifestPath)
      manifest <- parseManifest(key, manifestPath, manifestBytes)
      _ <-
        if manifest.provenance == input.provenance then Right(())
        else Left(invalid(key, manifestPath, "source provenance does not match the supplied input"))
      source <- readVerifiedFile(
        key,
        entry.resolve(CudaSourceFileName),
        manifest.cudaSourceSha256
      )
      _ <-
        if MessageDigest.isEqual(
            source,
            input.source.getBytes(StandardCharsets.UTF_8)
          )
        then Right(())
        else Left(NvrtcArtifactStoreSourceMismatch(key, entry))
      target = ComputeCapability(manifest.targetMajor, manifest.targetMinor)
      version = NvrtcVersion(manifest.nvrtcMajor, manifest.nvrtcMinor)
      compilerOptions = NvrtcCompileOptions.fromResolvedValues(
        manifest.compilerOptions
      )
      _ <- validateIdentity(
        key,
        manifestPath,
        input,
        version,
        target,
        compilerOptions,
        manifest.programName
      )
      ptx <- readVerifiedFile(
        key,
        entry.resolve(PtxFileName),
        manifest.ptxSha256
      )
      _ <-
        if ptx.nonEmpty then Right(())
        else
          Left(
            invalid(
              key,
              entry.resolve(PtxFileName),
              "PTX payload must not be empty"
            )
          )
      compileLog <- readVerifiedFile(
        key,
        entry.resolve(CompileLogFileName),
        manifest.compileLogSha256
      )
    yield NvrtcArtifact(
      input = input,
      ptx = IArray.unsafeFromArray(ptx),
      compileLog = String(compileLog, StandardCharsets.UTF_8),
      nvrtcVersion = version,
      target = target,
      compilerOptions = compilerOptions,
      programName = manifest.programName
    )

  private def writeArtifact(
      key: NvrtcCompilationKey,
      artifact: NvrtcArtifact,
      entry: Path
  ): Either[NvrtcArtifactStoreError, Unit] =
    val parent = entry.getParent
    var temporary: Option[Path] = None

    try
      Files.createDirectories(parent)
      if Files.exists(entry) then Right(())
      else
        val created = Files.createTempDirectory(parent, s".${key.toString}.")
        temporary = Some(created)
        writeEntry(created, key, artifact)
        try
          Files.move(created, entry, StandardCopyOption.ATOMIC_MOVE)
          temporary = None
        catch
          case _: IOException if Files.exists(entry) => ()
          case exception: IOException => throw exception
        Right(())
    catch
      case exception: AtomicMoveNotSupportedException =>
        Left(ioFailure("atomically publish artifact", entry, exception))
      case exception: IOException =>
        Left(ioFailure("write artifact", entry, exception))
      case exception: SecurityException =>
        Left(ioFailure("write artifact", entry, exception))
    finally
      temporary.foreach(deleteRecursively)

  private def readFile(
      key: NvrtcCompilationKey,
      path: Path
  ): Either[NvrtcArtifactStoreError, Array[Byte]] =
    try Right(Files.readAllBytes(path))
    catch
      case exception: IOException => Left(ioFailure("read artifact", path, exception))
      case exception: SecurityException => Left(ioFailure("read artifact", path, exception))

  private def readVerifiedFile(
      key: NvrtcCompilationKey,
      path: Path,
      expectedSha256: String
  ): Either[NvrtcArtifactStoreError, Array[Byte]] =
    readFile(key, path).flatMap { bytes =>
      val actualSha256 = sha256(bytes)
      if actualSha256 == expectedSha256 then Right(bytes)
      else
        Left(
          NvrtcArtifactStoreInvalidEntry(
            key,
            path,
            s"SHA-256 mismatch: expected $expectedSha256 but found $actualSha256"
          )
        )
    }

  private def validateIdentity(
      key: NvrtcCompilationKey,
      path: Path,
      input: NvrtcCompilationInput,
      version: NvrtcVersion,
      target: ComputeCapability,
      compilerOptions: NvrtcCompileOptions,
      programName: String
  ): Either[NvrtcArtifactStoreError, Unit] =
    val expectedOptions = NvrtcCompileOptions.resolve(
      input.compilerOptions,
      target
    )
    if compilerOptions != expectedOptions then
      Left(
        invalid(
          key,
          path,
          "resolved compiler options do not match the supplied input"
        )
      )
    else
      val derived = NvrtcCompilationKey.derive(
        input,
        target,
        version,
        programName
      )
      if derived == key then Right(())
      else
        Left(
          invalid(
            key,
            path,
            s"artifact metadata derives key $derived"
          )
        )

  private def writeEntry(
      temporary: Path,
      key: NvrtcCompilationKey,
      artifact: NvrtcArtifact
  ): Unit =
    val source = artifact.input.source.getBytes(StandardCharsets.UTF_8)
    val ptx = IArray.genericWrapArray(artifact.ptx).toArray
    val compileLog = artifact.compileLog.getBytes(StandardCharsets.UTF_8)

    Files.write(temporary.resolve(CudaSourceFileName), source)
    Files.write(temporary.resolve(PtxFileName), ptx)
    Files.write(temporary.resolve(CompileLogFileName), compileLog)
    Files.write(
      temporary.resolve(ManifestFileName),
      renderManifest(key, artifact, source, ptx, compileLog)
        .getBytes(StandardCharsets.UTF_8)
    )

  private def renderManifest(
      key: NvrtcCompilationKey,
      artifact: NvrtcArtifact,
      source: Array[Byte],
      ptx: Array[Byte],
      compileLog: Array[Byte]
  ): String =
    val provenanceFields = artifact.provenance match
      case NvrtcSourceProvenance.DslGenerated(codegenVersion) =>
        Vector(
          "source.provenance" -> "dsl",
          "source.codegen.version" -> codegenVersion.toString
        )
      case NvrtcSourceProvenance.CallerProvidedRaw =>
        Vector("source.provenance" -> "raw")
    val values = Vector(
      "schema" -> SchemaVersion.toString,
      "key" -> key.toString,
      "nvrtc.major" -> artifact.nvrtcVersion.major.toString,
      "nvrtc.minor" -> artifact.nvrtcVersion.minor.toString,
      "target.major" -> artifact.target.major.toString,
      "target.minor" -> artifact.target.minor.toString,
      "program.name" -> encode(artifact.programName),
      "options.count" -> artifact.compilerOptions.values.size.toString
    ) ++ provenanceFields ++ artifact.compilerOptions.values.zipWithIndex.map { case (option, index) =>
      s"option.$index" -> encode(option)
    } ++ Vector(
      "cuda.sha256" -> sha256(source),
      "ptx.sha256" -> sha256(ptx),
      "log.sha256" -> sha256(compileLog)
    )

    values.map { case (name, value) => s"$name=$value" }.mkString("", "\n", "\n")

  private def parseManifest(
      key: NvrtcCompilationKey,
      path: Path,
      bytes: Array[Byte]
  ): Either[NvrtcArtifactStoreError, Manifest] =
    parseFields(key, path, String(bytes, StandardCharsets.UTF_8)).flatMap {
      fields =>
        for
          schema <- integerField(key, path, fields, "schema")
          _ <-
            if schema == SchemaVersion then Right(())
            else Left(NvrtcArtifactStoreUnsupportedSchema(key, path, schema))
          provenance <- parseProvenance(key, path, fields)
          optionCount <- integerField(key, path, fields, "options.count")
          _ <-
            if optionCount > 0 then Right(())
            else Left(invalid(key, path, "options.count must be positive"))
          provenanceFields = provenance match
            case NvrtcSourceProvenance.DslGenerated(_) => Set("source.codegen.version")
            case NvrtcSourceProvenance.CallerProvidedRaw => Set.empty[String]
          expectedFields = RequiredFields ++ provenanceFields ++
            (0 until optionCount).map(index => s"option.$index").toSet
          _ <-
            if fields.keySet == expectedFields then Right(())
            else
              Left(
                invalid(
                  key,
                  path,
                  "manifest fields do not match schema " +
                    s"(expected ${expectedFields.toVector.sorted.mkString(", ")})"
                )
              )
          storedKey <- stringField(key, path, fields, "key")
          _ <-
            if storedKey == key.toString then Right(())
            else Left(invalid(key, path, s"manifest key is $storedKey"))
          nvrtcMajor <- integerField(key, path, fields, "nvrtc.major")
          nvrtcMinor <- integerField(key, path, fields, "nvrtc.minor")
          _ <-
            if nvrtcMajor >= 0 && nvrtcMinor >= 0 then Right(())
            else
              Left(
                invalid(
                  key,
                  path,
                  "NVRTC version components must not be negative"
                )
              )
          targetMajor <- integerField(key, path, fields, "target.major")
          targetMinor <- integerField(key, path, fields, "target.minor")
          _ <-
            if targetMajor > 0 && targetMinor >= 0 && targetMinor <= 9 then
              Right(())
            else
              Left(
                invalid(
                  key,
                  path,
                  "compute capability must have a positive major and a minor from 0 to 9"
                )
              )
          programNameEncoded <- stringField(key, path, fields, "program.name")
          programName <- decode(key, path, "program.name", programNameEncoded)
          _ <-
            if programName.nonEmpty && !programName.contains('\u0000') then
              Right(())
            else
              Left(
                invalid(
                  key,
                  path,
                  "program.name must be non-empty and null-free"
                )
              )
          options <- (0 until optionCount).foldLeft[
            Either[NvrtcArtifactStoreError, Vector[String]]
          ](Right(Vector.empty)) { (result, index) =>
            for
              values <- result
              encoded <- stringField(key, path, fields, s"option.$index")
              option <- decode(key, path, s"option.$index", encoded)
            yield values :+ option
          }
          _ <-
            if options.forall(option => option.nonEmpty && !option.contains('\u0000')) then
              Right(())
            else
              Left(
                invalid(
                  key,
                  path,
                  "resolved compiler options must be non-empty and null-free"
                )
              )
          cudaSourceSha256 <- sha256Field(key, path, fields, "cuda.sha256")
          ptxSha256 <- sha256Field(key, path, fields, "ptx.sha256")
          compileLogSha256 <- sha256Field(key, path, fields, "log.sha256")
        yield Manifest(
          provenance = provenance,
          nvrtcMajor = nvrtcMajor,
          nvrtcMinor = nvrtcMinor,
          targetMajor = targetMajor,
          targetMinor = targetMinor,
          programName = programName,
          compilerOptions = options,
          cudaSourceSha256 = cudaSourceSha256,
          ptxSha256 = ptxSha256,
          compileLogSha256 = compileLogSha256
        )
    }

  private def parseProvenance(
      key: NvrtcCompilationKey,
      path: Path,
      fields: Map[String, String]
  ): Either[NvrtcArtifactStoreError, NvrtcSourceProvenance] =
    stringField(key, path, fields, "source.provenance").flatMap {
      case "raw" => Right(NvrtcSourceProvenance.CallerProvidedRaw)
      case "dsl" =>
        integerField(key, path, fields, "source.codegen.version").flatMap { version =>
          if version > 0 then Right(NvrtcSourceProvenance.DslGenerated(version))
          else Left(invalid(key, path, "source.codegen.version must be positive"))
        }
      case other => Left(invalid(key, path, s"unknown source provenance: $other"))
    }

  private def parseFields(
      key: NvrtcCompilationKey,
      path: Path,
      contents: String
  ): Either[NvrtcArtifactStoreError, Map[String, String]] =
    if contents.isEmpty then Left(invalid(key, path, "manifest is empty"))
    else
      contents.linesIterator.foldLeft[
        Either[NvrtcArtifactStoreError, Map[String, String]]
      ](Right(Map.empty)) { (result, line) =>
        result.flatMap { fields =>
          val separator = line.indexOf('=')
          if separator <= 0 then
            Left(invalid(key, path, s"invalid manifest line: $line"))
          else
            val name = line.take(separator)
            val value = line.drop(separator + 1)
            if fields.contains(name) then
              Left(invalid(key, path, s"duplicate manifest field: $name"))
            else Right(fields.updated(name, value))
        }
      }

  private def stringField(
      key: NvrtcCompilationKey,
      path: Path,
      fields: Map[String, String],
      name: String
  ): Either[NvrtcArtifactStoreError, String] =
    fields.get(name).toRight(invalid(key, path, s"missing manifest field: $name"))

  private def integerField(
      key: NvrtcCompilationKey,
      path: Path,
      fields: Map[String, String],
      name: String
  ): Either[NvrtcArtifactStoreError, Int] =
    stringField(key, path, fields, name).flatMap { value =>
      value.toIntOption.toRight(
        invalid(key, path, s"manifest field $name must be an integer")
      )
    }

  private def sha256Field(
      key: NvrtcCompilationKey,
      path: Path,
      fields: Map[String, String],
      name: String
  ): Either[NvrtcArtifactStoreError, String] =
    stringField(key, path, fields, name).flatMap { value =>
      if Sha256Pattern.matches(value) then Right(value)
      else Left(invalid(key, path, s"manifest field $name is not a SHA-256 digest"))
    }

  private def decode(
      key: NvrtcCompilationKey,
      path: Path,
      name: String,
      value: String
  ): Either[NvrtcArtifactStoreError, String] =
    try Right(String(Base64.getUrlDecoder.decode(value), StandardCharsets.UTF_8))
    catch
      case _: IllegalArgumentException =>
        Left(invalid(key, path, s"manifest field $name is not URL-safe Base64"))

  private def encode(value: String): String =
    Base64.getUrlEncoder.withoutPadding.encodeToString(
      value.getBytes(StandardCharsets.UTF_8)
    )

  private def sha256(bytes: Array[Byte]): String =
    HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))

  private def invalid(
      key: NvrtcCompilationKey,
      path: Path,
      reason: String
  ): NvrtcArtifactStoreInvalidEntry =
    NvrtcArtifactStoreInvalidEntry(key, path, reason)

  private def ioFailure(
      operation: String,
      path: Path,
      exception: Exception
  ): NvrtcArtifactStoreIoFailure =
    NvrtcArtifactStoreIoFailure(operation, path, exception.getMessage)

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      Files.walkFileTree(
        path,
        new SimpleFileVisitor[Path]:
          override def visitFile(
              file: Path,
              attributes: BasicFileAttributes
          ): FileVisitResult =
            Files.delete(file)
            FileVisitResult.CONTINUE

          override def postVisitDirectory(
              directory: Path,
              exception: IOException
          ): FileVisitResult =
            if exception != null then throw exception
            Files.delete(directory)
            FileVisitResult.CONTINUE
      )

object NvrtcArtifactStore:
  private val SchemaVersion = 2
  private val ManifestFileName = "manifest"
  private val CudaSourceFileName = "source.cu"
  private val PtxFileName = "artifact.ptx"
  private val CompileLogFileName = "compile.log"
  private val Sha256Pattern = "[0-9a-f]{64}".r
  private val RequiredFields = Set(
    "schema",
    "key",
    "source.provenance",
    "nvrtc.major",
    "nvrtc.minor",
    "target.major",
    "target.minor",
    "program.name",
    "options.count",
    "cuda.sha256",
    "ptx.sha256",
    "log.sha256"
  )

  def apply(root: Path): NvrtcArtifactStore =
    new NvrtcArtifactStore(root)

  private final case class Manifest(
      provenance: NvrtcSourceProvenance,
      nvrtcMajor: Int,
      nvrtcMinor: Int,
      targetMajor: Int,
      targetMinor: Int,
      programName: String,
      compilerOptions: Vector[String],
      cudaSourceSha256: String,
      ptxSha256: String,
      compileLogSha256: String
  )
