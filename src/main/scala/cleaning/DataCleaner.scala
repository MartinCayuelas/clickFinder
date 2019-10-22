package cleaning

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import scala.annotation.tailrec

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
    val columns =  Seq("appOrSite", "bidFloor","media", "publisher", "os", "interests",  "user", "size", "type")
    dataFrame.select(columns.head, columns.tail: _*)
  }


  def cleanOS(dataFrame: DataFrame) : DataFrame = {
    val df_non_null = dataFrame.na.fill("UNKNOWN",Seq("os"))
    df_non_null.withColumn("os", when(col("os") === "ios", "iOS")
      .otherwise(when(col("os") === "Android", "android")
        .otherwise(when(col("os") === "Unknown", "UNKNOWN")
          .otherwise(when(col("os") === "other", "UNKNOWN")
            .otherwise(when(col("os") === "Windows Mobile OS", "WindowsPhone")
              .otherwise(when(col("os") ===  "WindowsMobile", "WindowsPhone")
                .otherwise(when(col("os") ===  "windows", "WindowsPhone")
                  .otherwise(when(col("os") === "Windows Phone OS"  , "WindowsPhone")
                    .otherwise(col("os")))))))))
    )
  }

  def cleanBidFloor(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill(0,Seq("bidFloor"))
  }

  /**
   * Applies all the cleaning method to the dataframe in parameter
   * @param dataFrame
   * @return
   */
  def clean(dataFrame: DataFrame): DataFrame = {
    val methods: Seq[DataFrame => DataFrame] =  Seq(cleanOS, cleanBidFloor)
    @tailrec
    def applyMethods(methods: Seq[DataFrame => DataFrame], res: DataFrame): DataFrame = {
      methods match {
        case Seq() => res
        case _ => applyMethods(methods.tail, methods.head(res))
      }
    }
    applyMethods(methods, dataFrame)
  }

  /**
   * TODO - Pass an array of string corresponding to the columns to select
   * @return The dataFrame with clean entries
   */
  def retrieveDataFrame(): DataFrame = {
    val df = selectData(readDataFrame())
    clean(df)
  }


  /**
   * TEST CASE
   */
  def main(args: Array[String]) {
    val df = selectData(readDataFrame())
    val res = clean(df)
    res.printSchema
    res.select("os").distinct.show()
    res.select("bidFloor").distinct.show()
    spark.stop()
  }
}
