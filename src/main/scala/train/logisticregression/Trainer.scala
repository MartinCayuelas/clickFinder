package train.logisticregression

import eval.Evaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import utils.Tools

object Trainer {

  def main(args: Array[String]) {

    val spark = SparkSession
       .builder()
       .appName("ClickFinder")
       .config("spark.master", "local")
       .getOrCreate()

    import spark.implicits._

    val raw_data = Tools.retrieveDataFrameCleaned()
    //val raw_data = Tools.limitDataFrame(Tools.retrieveDataFrameCleaned(), 100000)


    val featuresCols = Array("appOrSite", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val data = assembler.transform(raw_data).select($"features", $"label")
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 42)

    val balanced_dataset = Tools.balanceDataset(trainingData)
    balanced_dataset.show(20)

    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("classWeightCol")
      //.setFamily("binomial")
      .setMaxIter(30)
      .setRegParam(0)
      .setThreshold(0.55)

    // Fit the model for Logistic Regression
    val balancedLR = lr.fit(balanced_dataset)
    //balancedLR.write.overwrite().save("models/balancedModel")

    // Get predictions
    val predictionsBalancedLR = balancedLR.transform(testData)

    //Evaluator for the classification
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val accuracyBLR = evaluator.evaluate(predictionsBalancedLR)
    println("areaUnderROC: " + accuracyBLR)

    Evaluator.retrieveMetrics(predictionsBalancedLR, "results/resultLR.txt")

    println(s"Intercept: ${balancedLR.intercept}")

    predictionsBalancedLR.show(20)
    Tools.saveDataFrameToCsv(predictionsBalancedLR.select($"label", $"prediction"), "predictionBLR")
    spark.stop()
  }
}
