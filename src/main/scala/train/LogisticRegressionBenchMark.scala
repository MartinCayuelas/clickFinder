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

  def regression(balanced_dataset: DataFrame, testData: DataFrame, maxIter: Int, regParam: Double, threshold: Double) = {
    writeFile(s"--------------------------------------------------------------------\nPARAMS:\n maxIter: $maxIter \n regParam $regParam \n threshold $threshold")

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
    Evaluator.retrieveMetrics(predictionsBalancedLR)
    println(s"Intercept: ${balancedLR.intercept}")

    predictionsBalancedLR.show(10)
    writeFile("areaUnderROC: " + accuracyBLR +"\n")
    //Tools.saveDataFrameToCsv(predictionsBalancedLR.select($"label", $"prediction"), "predictionBLR")

  }

  def writeFile(s: String): Unit = {
    val file = new File("resultLR.txt")
    val bw = new BufferedWriter(new FileWriter(file, true))
    bw.write(s)
    bw.write("\n")
    bw.close()
  }
}