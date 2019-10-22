package cleaning

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object DataCleaner {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("Click Finder")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  def readDataFrame(): DataFrame = {
    val data = spark.read.format("json")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("data-students.json")
    data
  }

  def selectData(dataFrame: DataFrame): DataFrame = {
    val columns =  Seq("appOrSite", "bidFloor", "interests", "media", "publisher", "user", "size", "type")
    dataFrame.select( columns.head, columns.tail: _*)
  }


  def cleanOS(dataFrame: DataFrame) : DataFrame = {
    dataFrame.na.fill("UNKNOWN",Seq("os"))
    dataFrame.withColumn("os", when(col("os") === "ios", "iOS")
      .otherwise(when(col("os") === "Android", "android"))
      .otherwise(when(col("os") === "Windows Phone OS", "WindowsPhone"))
      .otherwise(col("make"))
    )
  }

  def clean(dataFrame: DataFrame): DataFrame = {
    cleanOS(dataFrame)
  }

  def retrieveDataFrame(dataFrame: DataFrame): DataFrame = {
    val df = selectData(readDataFrame())
    clean(df)
  }

  def main(args: Array[String]) {
    val df = selectData(readDataFrame())
    val res = clean(df)
    res.printSchema
    spark.stop()
  }
}
