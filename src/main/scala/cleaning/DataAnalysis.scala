package cleaning

import cleaning.DataCleaner.{clean, readDataFrame}
import org.apache.spark.sql.SparkSession

object DataAnalysis {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("Click Finder")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  def main(args: Array[String]) {
    /*
    val logFile = "README.md" // Should be some file on your system
    val logData = spark.read.textFile(logFile).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println(s"Lines with a: $numAs, Lines with b: $numBs")
*/

    val users = spark.read.format("json")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("data-students.json")

    //val df = spark.read.json("data-students.json")
    val df = clean(readDataFrame())
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

    /*
    spark.sql("SELECT exchange FROM user GROUP BY exchange").show()
    spark.sql("SELECT impid FROM user GROUP BY impid").show()
    spark.sql("SELECT media FROM user GROUP BY media").show()
    spark.sql("SELECT network FROM user GROUP BY network").show()
    spark.sql("SELECT publisher FROM user GROUP BY publisher").show()
    spark.sql("SELECT element FROM user GROUP BY element").show()
    spark.sql("SELECT city FROM user GROUP BY city").show()
    */

    /*
    users.printSchema
    users.select("appOrSite").distinct.show()
    users.select("bidfloor").distinct.show()
    users.select("city").distinct.show()
    users.select("exchange").distinct.show()
    users.select("impid").distinct.show()
    users.select("interests").distinct.show()
    */
    spark.stop()
  }
}
