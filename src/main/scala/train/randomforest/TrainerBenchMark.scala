package train.randomforest

import eval.Evaluator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.DataFrame
import utils.Tools

object TrainerBenchMark {

  /**
   * Run a random forest with parameters
   * @param balanced_dataset : dataset to train
   * @param testData : data for test the train
   * @param impurity : type of the method
   * @param maxDepth : max number of depth
   * @param numTrees : number of trees
   * @param maxBins : max number of bins
   */
  def randomForest(balanced_dataset: DataFrame, testData: DataFrame, impurity: String, maxDepth: Int, numTrees: Int, maxBins: Int) = {

    Tools.writeFile(s"--------------------------------------------------------------------\nPARAMS:\n impurity: $impurity \n maxDepth $maxDepth \n numTrees $numTrees \n maxBins $maxBins", "results/resultRF.txt")



    val randomForestClassifier = new RandomForestClassifier()
      .setImpurity(impurity)
      .setMaxDepth(maxDepth)
      .setNumTrees(numTrees)
      .setMaxBins(maxBins)
      .setFeatureSubsetStrategy("auto")
      .setSeed(42)

    // Fit the model for Random Forest
    val balancedRF = randomForestClassifier.fit(balanced_dataset)

    // Get predictions
    val predictionsBalancedRF = balancedRF.transform(testData)

    //Evaluator for the classification
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val accuracyBRF = evaluator.evaluate(predictionsBalancedRF)

    println("areaUnderROC: " + accuracyBRF)
    Evaluator.retrieveMetrics(predictionsBalancedRF, "results/resultRF.txt")

    predictionsBalancedRF.show(10)
    Tools.writeFile("areaUnderROC: " + accuracyBRF +"\n", "results/resultRF.txt")

  }

}
