package utils

import clean.DataCleaner.{clean, readDataFrame, selectData}
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
      .save(s"data/$name")
  }

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

}
