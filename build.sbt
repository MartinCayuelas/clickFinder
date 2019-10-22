name := "clickHandlers"

version := "0.1"

scalaVersion := "2.12.8"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-streaming" % "2.4.3" ,
  "org.apache.spark" %% "spark-sql" % "2.4.3",
  "org.apache.spark" %% "spark-mllib" % "2.4.3",

  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "org.scalacheck" %% "scalacheck" % "1.13.4" % "test"
)