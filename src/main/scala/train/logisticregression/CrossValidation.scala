package train.logisticregression

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import utils.Tools

object CrossValidation {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    val raw_data = Tools.limitDataFrame(Tools.retrieveDataFrameCleaned(), 100000)

    val featuresCols = Array("appOrSite", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val data = assembler.transform(raw_data).select( $"features", $"label")
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 42)
    val balanced_dataset = Tools.balanceDataset(trainingData)

    //Parameters to change
    val maxIterTab = Array(120)
    val regParamsTab = Array(0.03)
    val thresholdTab = Array(0.54)

    maxIterTab.foreach(maxIter =>
      regParamsTab.foreach(regParams =>
        thresholdTab.foreach(threshold =>
          TrainerBenchMark.logisticRegression(balanced_dataset, testData, maxIter, regParams, threshold)
        )
      )
    )

    spark.stop()
  }
}
