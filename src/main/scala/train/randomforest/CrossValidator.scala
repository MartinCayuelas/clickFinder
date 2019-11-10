package train.randomforest

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import utils.Tools

object CrossValidator {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    //val raw_data = Tools.retrieveDataFrameCleaned()
    val raw_data = Tools.limitDataFrame(Tools.retrieveDataFrameCleaned(), 100000)

    val featuresCols = Array("appOrSite", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val data = assembler.transform(raw_data).select( $"features", $"label")
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 42)
    val balanced_dataset = Tools.balanceDataset(trainingData)

    //Parameters to change
    val impurityTab = Array("gini")
    val maxDepthTab = Array(3)
    val numTreeTab = Array(20)
    val maxBinsTab = Array(31)

    impurityTab.foreach(impurity =>
      maxDepthTab.foreach(maxDepth =>
        numTreeTab.foreach(numTree =>
          maxBinsTab.foreach(maxBins =>
            TrainerBenchMark.randomForest(balanced_dataset, testData, impurity, maxDepth, numTree, maxBins)
          )
        )
      )
    )

    spark.stop()
  }

}
