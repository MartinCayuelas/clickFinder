package train

import java.io.{BufferedWriter, File, FileWriter}

import eval.Evaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import utils.Tools


object LogisticRegressionBenchMark {

  def logisticRegression(balanced_dataset: DataFrame, testData: DataFrame, maxIter: Int, regParam: Double, threshold: Double) = {

    Tools.writeFile(s"--------------------------------------------------------------------\nPARAMS:\n maxIter: $maxIter \n regParam $regParam \n threshold $threshold", "results/resultLR.txt")

    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("classWeightCol")
      .setFamily("binomial")
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setThreshold(threshold)

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

    predictionsBalancedLR.show(10)
    Tools.writeFile("areaUnderROC: " + accuracyBLR +"\n", "results/resultLR.txt")
    //Tools.saveDataFrameToCsv(predictionsBalancedLR.select($"label", $"prediction"), "predictionBLR")

  }

}