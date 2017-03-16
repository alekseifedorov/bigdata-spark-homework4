name := "homework1"

version := "0.0.1"

scalaVersion := "2.11.0"

logBuffered in Test := false

// additional libraries
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.2",
  "org.apache.spark" %% "spark-mllib" % "1.6.2"
)
