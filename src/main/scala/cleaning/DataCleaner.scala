package cleaning

import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.regexp_replace

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
    //val columns =  Seq("appOrSite", "bidFloor","media", "publisher", "os", "interests", "size", "type", "network")
    val columns =  Seq("appOrSite", "bidFloor", "os", "label")
    dataFrame.select(columns.head, columns.tail: _*)
  }

  def cleanLabel(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("label",dataFrame("label").cast("Int"))
  }

  def cleanOS(dataFrame: DataFrame) : DataFrame = {
    val df_non_null = dataFrame.na.fill("UNKNOWN",Seq("os"))
    df_non_null.withColumn("os", when(col("os") === "ios" || col("os") === "iOS", "3")
      .otherwise(when(col("os") === "Android" || col("os") === "android", "2")
        .otherwise(when(col("os") === "Unknown" || col("os") === "UNKNOWN", "0")
          .otherwise(when(col("os") === "other", "1")
            .otherwise(when(col("os") === "Windows Mobile OS" || col("os") === "WindowsMobile" || col("os") === "windows" || col("os") === "Windows Phone OS" || col("os") === "WindowsPhone", "4")
              .otherwise(when(col("os") === "blackberry", "5")
                .otherwise(when(col("os") === "Rim", "6")
                  .otherwise(when(col("os") === "WebOS", "7")
                    .otherwise(when(col("os") === "Symbian", "8")
                      .otherwise(when(col("os") === "Bada", "9")
                    .otherwise(col("os")))))))))))
    )
  }

  def cleanAppOrSite(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("appOrSite", when(col("appOrSite") === "app", "1")
      .otherwise(when(col("appOrSite") === "site", "2"))
    )
  }

  def cleanBidFloor(dataFrame: DataFrame) : DataFrame = {
    dataFrame.na.fill(0,Seq("bidFloor"))
  }

  def cleanNetwork(dataFrame: DataFrame) : DataFrame = {
    val df_non_null = dataFrame.na.fill("UNKNOWN",Seq("network"))
    df_non_null.withColumn("network", when(col("network") === "other", "UNKNOWN"))
  }

  def cleanSize(dataFrame: DataFrame) : DataFrame = {
    dataFrame.na.fill("UNKNOWN",Seq("size"))
  }

  /**
   * Removes the sub-categories for the interests column.
   * @param dataFrame
   * @return the dataFrame with the column interests cleaned
   */
  def cleanInterests(dataFrame: DataFrame): DataFrame = {
    val df_without_sub = dataFrame.withColumn("interests", regexp_replace(dataFrame("interests"), "-[0-9]", ""))
    df_without_sub.na.fill("UNKNOWN",Seq("interests"))
  }

  /**
   * Applies all the cleaning method to the dataframe in parameter
   * @param dataFrame
   * @return
   */
  def clean(dataFrame: DataFrame): DataFrame = {
    //val methods: Seq[DataFrame => DataFrame] =  Seq(cleanOS, cleanBidFloor, cleanInterests, cleanNetwork, cleanSize)
    val methods: Seq[DataFrame => DataFrame] =  Seq(cleanOS, cleanBidFloor, cleanAppOrSite, cleanLabel)
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
   * Limit the number of entries for a dataframe
   * @param df : dataframe to limit
   * @param size : number of entries wanted
   * @return a new dataframe with less entries
   */
  def limitDataFrame(df: DataFrame, size: Int): DataFrame = df.limit(size)

  /**
   * Save a dataframe in txt file
   * @param df : dataframe to save in a file
   */
  def saveDataFrameToCsv(df: DataFrame): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    df.coalesce(1)
      //.withColumn("size", stringify($"size"))
      .write.format("com.databricks.spark.csv")
      .option("sep", ";")
      .option("header", "true")
      .save("data/data-student")
  }

  def stringify(c: Column) = concat(lit("["), concat_ws(",", c), lit("]"))
  
  /**
   * TEST CASE
   */
  def main(args: Array[String]) {
    val df = selectData(readDataFrame())
    val res = clean(df)
    res.printSchema
    println("DataFrame size : " + res.count())
    //Limit the df and save in files
    val limitedDf = limitDataFrame(res, 1000)
    saveDataFrameToCsv(limitedDf)
    //res.select("os").distinct.show()
    //res.select("bidFloor").distinct.show()
    spark.stop()
  }
}
