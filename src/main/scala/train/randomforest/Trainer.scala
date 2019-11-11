package train.randomforest

import clean.DataCleaner
import eval.Evaluator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import utils.Tools

object Trainer {

  def train(spark: SparkSession, path: String, name: String) {
    /*val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()
      */
    import spark.implicits._
    //val raw_data = Tools.retrieveDataFrameCleaned()
    val raw_data = DataCleaner.clean(Tools.selectData(Tools.readDataFrame(path)))
    //val raw_data = Tools.limitDataFrame(Tools.retrieveDataFrameCleaned(), 100000)


    val featuresCols = Array("appOrSite", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val data = assembler.transform(raw_data).select( $"features", $"label")
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 42)
    val randomForestClassifier = new RandomForestClassifier()
      .setImpurity("entropy")
      .setMaxDepth(10)
      .setNumTrees(50)
      .setMaxBins(110)
      .setFeatureSubsetStrategy("auto")
      .setSeed(42)

    val balancedRF = randomForestClassifier.fit(trainingData)
    balancedRF.write.overwrite().save(s"models/${name}")

    val predictionsBalancedRF = balancedRF.transform(testData)

    //Evaluator for the classification
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")

    val accuracyBRF = evaluator.evaluate(predictionsBalancedRF)
    println("areaUnderROC: " + accuracyBRF)
    predictionsBalancedRF.show(50)
    Evaluator.retrieveMetrics(predictionsBalancedRF, "results/resultRF.txt")

    spark.stop()
  }
}
