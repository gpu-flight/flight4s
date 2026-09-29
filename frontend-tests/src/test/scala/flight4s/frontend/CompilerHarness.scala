package flight4s.frontend

import java.io.File
import java.net.URLClassLoader
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.*
import dotty.tools.dotc.Main
import dotty.tools.dotc.core.Contexts.Context
import dotty.tools.dotc.reporting.{Diagnostic, Reporter}

private[frontend] object CompilerHarness:
  final case class Result(errors: Vector[String], classes: Path)

  def compile(directory: Path, name: String, source: String,
      experimental: Boolean = false, dependencies: Seq[Path] = Nil): Result =
    val sourceFile = directory.resolve(s"$name.scala")
    Files.writeString(sourceFile, source, UTF_8)
    val classes = Files.createDirectories(directory.resolve(s"$name-classes"))
    val errors = ArrayBuffer.empty[String]
    val reporter = new Reporter:
      override def doReport(diagnostic: Diagnostic)(using Context): Unit =
        if diagnostic.isInstanceOf[Diagnostic.Error] then errors += diagnostic.message
    val classpath = (sys.props("java.class.path") +: dependencies.map(_.toString)).mkString(File.pathSeparator)
    val options = Vector("-classpath", classpath, "-d", classes.toString,
      "-Xcheck-macros", "-Ycheck:all", "-color:never") ++
      (if experimental then Vector("-experimental") else Vector.empty) :+ sourceFile.toString
    Main.process(options.toArray, reporter)
    Result(errors.toVector, classes)

  def withDirectory[A](body: Path => A): A =
    val directory = Files.createTempDirectory("flight4s-frontend-")
    try body(directory)
    finally
      val paths = Files.walk(directory)
      try paths.iterator().asScala.toVector.reverse.foreach(Files.delete)
      finally paths.close()

  def withClasses[A](paths: Seq[Path])(body: URLClassLoader => A): A =
    val loader = URLClassLoader(paths.map(_.toUri.toURL).toArray, getClass.getClassLoader)
    try body(loader)
    finally loader.close()
