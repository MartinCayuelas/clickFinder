package utils

import java.io.{BufferedWriter, File, FileWriter}
import java.nio.file.Paths

import clean.DataCleaner.{clean,spark}
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.{concat, concat_ws, lit, udf}

object Tools {

  /**
   * Retrieve and clean a dataframe
   * @param pathToFile : path of the file to retrieve
   * @return a cleaned DataFrame
   */
  def retrieveDataFrameCleaned(pathToFile: String ="data-students.json"): DataFrame = {
    val df = selectData(readDataFrame(pathToFile))
    clean(df)
  }

  def readDataFrame(pathToFile: String): DataFrame = {
    val data = spark.read.format("json")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(Paths.get(pathToFile).toAbsolutePath.toString)
    data
  }

  /**
   * Select the necessary columns
   * @param dataFrame
   * @return
   */
  def selectData(dataFrame: DataFrame): DataFrame = {
    val columns = Seq("appOrSite", "bidFloor", "timestamp", "os", "size", "label", "type", "interests", "media", "exchange")
    dataFrame.select(columns.head, columns.tail: _*)
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
   * @param name : name for the output csv file
   */
  def saveDataFrameToCsv(df: DataFrame, name: String): Unit = {
    df.repartition(1).coalesce(1)
      .write
      .mode ("overwrite")
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save(s"$name")
  }

  /**
   * Add a new column classWeightCol to the Dataset with the weight of each class.
   * @param dataset
   * @return
   */
  def balanceDataset(dataset: DataFrame): DataFrame = {
    // Re-balancing (weighting) of records to be used in the logistic loss objective function
    val numNegatives = dataset.filter(dataset("label") === 0).count
    val datasetSize = dataset.count
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize

    val calculateWeights = udf { d: Double =>
      if (d == 0.0) {
        1 * balancingRatio
      }
      else {
        1 * (1.0 - balancingRatio)
      }
    }
    val weightedDataset = dataset.withColumn("classWeightCol", calculateWeights(dataset("label")))
    weightedDataset
  }

  def stringify(c: Column): Column = concat(lit("["), concat_ws(",", c), lit("]"))

  def writeFile(s: String, filePath: String): Unit = {
    val file = new File(filePath)
    val bw = new BufferedWriter(new FileWriter(file, true))
    bw.write(s)
    bw.write("\n")
    bw.close()
  }

}
