package clean

import clean.DataCleaner.{clean, readDataFrame}
import org.apache.spark.sql.SparkSession

object DataAnalysis {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("Click Finder")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  def main(args: Array[String]) {

    val users = spark.read.format("json")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("data-students.json")

    val df = clean(readDataFrame("data-students.json"))
    df.printSchema()
    df.createOrReplaceTempView("user")

    spark.sql("SELECT exchange FROM user GROUP BY exchange").show()
    spark.sql("SELECT publisher FROM user GROUP BY publisher").show()
    spark.sql("SELECT media FROM user GROUP BY media").show()
    spark.sql("SELECT appOrSite FROM user GROUP BY appOrSite").show()
    spark.sql("SELECT bidfloor FROM user GROUP BY bidfloor").show()
    spark.sql("SELECT timestamp FROM user GROUP BY timestamp").show()
    spark.sql("SELECT type FROM user GROUP BY type").show()
    spark.sql("SELECT size FROM user GROUP BY size").show()
    spark.sql("SELECT os FROM user GROUP BY os").show()
    spark.sql("SELECT interests FROM user GROUP BY interests").show()
    spark.sql("SELECT label FROM user GROUP BY label").show()

    spark.stop()
  }
}
