package analysis

import org.apache.spark.sql.SparkSession

object SimpleApp {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("Click prediction")
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

    /*
    val df = spark.read.json("data-students.json")
    df.createOrReplaceTempView("user")
    val sqlDF = spark.sql("SELECT distinct appOrSite FROM user")
    sqlDF.show()
    */

    users.printSchema
    users.select("appOrSite").distinct.show()
    users.select("bidfloor").distinct.show()
    users.select("city").distinct.show()
    users.select("exchange").distinct.show()
    users.select("impid").distinct.show()
    users.select("interests").distinct.show()




    spark.stop()
  }
}
