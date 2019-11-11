package train.logisticregression

import eval.Evaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.DataFrame
import utils.Tools

object TrainerBenchMark {

  /**
   * Run a logistic regression with parameters
   * @param balanced_dataset : the dataset to train
   * @param testData : the data for test the train
   * @param maxIter : max number of iterations
   * @param regParam : regularization parameter
   * @param threshold : limit between the 2 labels
   */
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

    Tools.writeFile("areaUnderROC: " + accuracyBLR +"\n", "results/resultLR.txt")

  }

}
