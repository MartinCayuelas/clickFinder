package cleaning

import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.regexp_replace

import scala.annotation.tailrec
import scala.util.matching.Regex


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
    val columns = Seq("appOrSite", "bidFloor", "timestamp", "os", "size", "label", "type", "interests", "media", "exchange")
    dataFrame.select(columns.head, columns.tail: _*)
  }

  def cleanLabel(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("label", dataFrame("label").cast("Int"))
  }

  /**
   * 1 for Android, 2 for iOS, 3 for Windows, 0 for others
   *
   * @param dataFrame
   * @return
   */
  def cleanOS(dataFrame: DataFrame): DataFrame = {
    val df_non_null = dataFrame.na.fill(0, Seq("os"))
    df_non_null.withColumn("os", when(col("os").contains("android"), 1)
      .when(col("os").contains("Android"), 1)
      .when(col("os").contains("ios"), 2)
      .when(col("os").contains("iOS"), 2)
      .when(col("os").contains("windows"), 3)
      .when(col("os").contains("Windows Mobile OS"), 3)
      .when(col("os").contains("WindowsMobile"), 3)
      .when(col("os").contains("Windows Phone OS"), 3)
      .when(col("os").contains("WindowsPhone"), 3)
      .otherwise(0)
    )
  }

  /**
   * 0 to 4 for exchange
   *
   * @param dataFrame
   * @return
   */
  def cleanExchange(dataFrame: DataFrame): DataFrame = {
    val df_non_null = dataFrame.na.fill(0, Seq("exchange"))
    df_non_null.withColumn("exchange", when(col("exchange").contains("f8dd61fb7d4ebfa62cd6acceae3f5c69"), 1)
      .when(col("exchange").contains("c7a327a5027c1c4de094b0a9f33afad6"), 2)
      .when(col("exchange").contains("46135ae0b4946b5f2f74274e5618e697"), 3)
      .when(col("exchange").contains("fe86ac12a6d9ccaa8a2be14a80ace2f8"), 4)
      .otherwise(0)
    )
  }

  /**
   * 0 to 4 for exchange
   *
   * @param dataFrame
   * @return
   */
  def cleanMedia(dataFrame: DataFrame): DataFrame = {
    val df_non_null = dataFrame.na.fill(0, Seq("media"))
    df_non_null.withColumn("media", when(col("media").contains("d476955e1ffb87c18490e87b235e48e7"), 1)
      .when(col("media").contains("343bc308e60156fb39cd2af57337a958"), 2)
      .otherwise(0)
    )
  }

  /**
   * 1 for App, 2 for Site, 0 otherwise
   *
   * @param dataFrame
   * @return
   */
  def cleanAppOrSite(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("appOrSite",
      when(col("appOrSite") === "app", 1)
        .when(col("appOrSite") === "site", 2)
        .otherwise(0)
    )
  }

  /**
   * Replace null values with the average bidfloor
   *
   * @param dataFrame
   * @return
   */
  def cleanBidFloor(dataFrame: DataFrame): DataFrame = {
    val averageBidFloor = dataFrame.select(mean(dataFrame("bidfloor"))).first()(0).asInstanceOf[Double]
    dataFrame.na.fill(averageBidFloor, Seq("bidFloor"))
  }

  /**
   * Replace null values with the average timestamp
   *
   * @param dataFrame
   * @return
   */
  def cleanTimestamp(dataFrame: DataFrame): DataFrame = {
    val averageTimestamp = dataFrame.select(mean(dataFrame("timestamp"))).first()(0).asInstanceOf[Double]
    dataFrame.na.fill(averageTimestamp, Seq("timestamp"))
  }

  /**
   * Replace null values 5 and CLICK with 4
   *
   * @param dataFrame
   * @return
   */
  def cleanType(dataFrame: DataFrame): DataFrame = {
    val cleanDF = dataFrame.withColumn("type",
      when(col("type") === "CLICK", 4)
        .when(col("type") === "0", 0)
        .when(col("type") === "1", 1)
        .when(col("type") === "2", 2)
        .when(col("type") === "3", 3))
    cleanDF.na.fill(5, Seq("type"))
  }


  def cleanNetwork(dataFrame: DataFrame): DataFrame = {
    val df_non_null = dataFrame.na.fill("UNKNOWN", Seq("network"))
    df_non_null.withColumn("network", when(col("network") === "other", "UNKNOWN"))
  }

  /**
   * Replace the size with the Screen Type:
   * 0 if null
   * 1 for a square screen (L == H)
   * 2 for horizontal (L > H)
   * 3 for vertical (L < H)
   *
   * @param dataFrame
   * @return
   */
  def cleanSize(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("size",
      when(col("size").isNotNull && col("size")(0).equals(col("size")(1)), 1)
        .when(col("size").isNotNull && col("size")(0) > col("size")(1), 2)
        .when(col("size").isNotNull && col("size")(0) < col("size")(1), 3)
        .otherwise(0)
    )
  }

  /**
   * Removes the sub-categories for the interests column.
   * TODO - UNKNOWN ?
   *
   * @param dataFrame
   * @return the dataFrame with the column interests cleaned
   */
  def cleanInterests(dataFrame: DataFrame): DataFrame = {
    val res = dataFrame.withColumn("interests", regexp_replace(dataFrame("interests"), "IAB|-[0-9]*", ""))
    var df_non_null = res.na.fill("UNKNOWN", Seq("interests"))
    val sqlfunc = (interestNumer: String) => {
      val expression = new Regex("(\\A|[^0-9])4([^0-9]|\\z)")
      expression.findAllIn("4").length match {
        case 0 => col("interests").contains("vvvv").cast("Int")
        case _ => col("interests").contains(interestNumer).cast("Int")
      }
    }
    for (i <- 1 to 26) df_non_null = df_non_null.withColumn("IAB" + i.toString, sqlfunc(i.toString))
    val res2 = df_non_null
    res2.printSchema()
    res2.show(10)
    res2
  }

  /**
   * Applies all the cleaning method to the dataframe in parameter
   *
   * @param dataFrame
   * @return
   */
  def clean(dataFrame: DataFrame): DataFrame = {
    val methods: Seq[DataFrame => DataFrame] = Seq(cleanOS, cleanBidFloor, cleanAppOrSite, cleanLabel, cleanTimestamp, cleanSize, cleanType, cleanInterests, cleanExchange, cleanMedia)

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
   *
   * @return The dataFrame with clean entries
   */
  def retrieveDataFrame(): DataFrame = {
    val df = selectData(readDataFrame())
    clean(df)

  }

  /**
   * Limit the number of entries for a dataframe
   * @param df   : dataframe to limit
   * @param size : number of entries wanted
   * @return a new dataframe with less entries
   */
  def limitDataFrame(df: DataFrame, size: Int): DataFrame = /*df.sample(false, 0.0001, 12345)
*/ df.limit(size)

  /**
   * Save a dataframe in txt file
   * @param df : dataframe to save in a file
   */
  def saveDataFrameToCsv(df: DataFrame, name: String): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    df.coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("sep", ";")
      .option("header", "true")
      .save(s"data/$name")
  }

  def stringify(c: Column) = concat(lit("["), concat_ws(",", c), lit("]"))



  /**
   * TEST CASE
   */
  def main(args: Array[String]) {
    val df = clean(selectData(readDataFrame()))
    saveDataFrameToCsv(df, "cleaned")
    //df.write.json("result")
    /*
    val df = selectData(readDataFrame())
    val res = clean(df)
    res.printSchema
    res.select("os").distinct.show()
    res.select("bidFloor").distinct.show()
     */
    println("DataFrame size : " + df.count())
    //Limit the df and save in files
    val limitedDf = limitDataFrame(df, 100)
    //saveDataFrameToCsv(limitedDf)
    //res.select("os").distinct.show()
    //res.select("bidFloor").distinct.show()
    spark.stop()
  }
}
