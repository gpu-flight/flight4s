ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.8.1"
ThisBuild / organization := "io.github.gpu-flight"
ThisBuild / homepage := Some(url("https://github.com/gpu-flight/flight4s"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/gpu-flight/flight4s"),
    "scm:git:https://github.com/gpu-flight/flight4s.git"
  )
)

lazy val root = (project in file("."))
  .aggregate(core, runtime, examples, frontend, frontendTests)
  .settings(
    name := "flight4s-root",
    publish / skip := true
  )

lazy val core = (project in file("core"))
  .settings(
    name := "flight4s-core",
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test
  )

lazy val runtime = (project in file("runtime"))
  .dependsOn(core)
  .settings(
    name := "flight4s-runtime",
    // A native library cannot be reloaded into sbt's next test classloader.
    Test / fork := sys.props.contains("flight4s.cuda.native.path"),
    Test / javaOptions ++= sys.props.get("flight4s.cuda.native.path")
      .map(path => s"-Dflight4s.cuda.native.path=$path").toSeq,
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test
  )

lazy val examples = (project in file("examples"))
  .dependsOn(runtime)
  .settings(
    name := "flight4s-examples",
    publish / skip := true,
    // JNI libraries cannot be loaded by separate sbt test/run classloaders in one JVM.
    fork := true,
    javaOptions ++= sys.props.get("flight4s.cuda.native.path")
      .map(path => s"-Dflight4s.cuda.native.path=$path").toSeq,
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test
  )

lazy val frontend = (project in file("frontend"))
  .dependsOn(core)
  .settings(
    name := "flight4s-frontend",
    scalacOptions ++= Seq("-Xcheck-macros", "-Ycheck:all")
  )

lazy val frontendTests = (project in file("frontend-tests"))
  .dependsOn(frontend, runtime)
  .settings(
    name := "flight4s-frontend-tests",
    publish / skip := true,
    Compile / scalacOptions ++= Seq("-Xcheck-macros", "-Ycheck:all"),
    Test / scalacOptions ++= Seq("-Xcheck-macros", "-Ycheck:all"),
    Test / fork := true,
    Test / javaOptions ++= sys.props.get("flight4s.cuda.native.path")
      .map(path => s"-Dflight4s.cuda.native.path=$path").toSeq,
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "org.scala-lang" %% "scala3-compiler" % scalaVersion.value % Test
    )
  )
