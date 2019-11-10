package train

import java.io.{BufferedWriter, File, FileWriter}

import eval.Evaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import utils.Tools


object LogisticRegressionBenchMark {

  /*def logisticRegression(balanced_dataset: DataFrame, testData: DataFrame, maxIter: Int, regParam: Double, threshold: Double) = {

    Tools.writeFile(s"--------------------------------------------------------------------\nPARAMS:\n maxIter: $maxIter \n regParam $regParam \n threshold $threshold", "results/resultLR.txt")

    val randomForestClassifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(3)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setSeed(42)

    // evaluate model with area under ROC
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")


    val stages = Array(assembler, randomForestClassifier)

    // build pipeline
    val pipeline = new Pipeline().setStages(stages)
    val pipelineModel = pipeline.fit(trainingData)

    // test model with test data
    val pipelinePredictionDf = pipelineModel.transform(testData)
    pipelinePredictionDf.show(10)
    val pipelineAccuracy = evaluator.evaluate(pipelinePredictionDf)
    println(pipelineAccuracy)

    val paramGrid = new ParamGridBuilder()
      /*
  .addGrid(randomForestClassifier.maxBins, Array(100, 90, 80))
  .addGrid(randomForestClassifier.numTrees, Array(40, 35, 30))
  .addGrid(randomForestClassifier.maxDepth, Array(7, 8, 9))
  .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
*/
      .addGrid(randomForestClassifier.maxBins, Array(31))
      .addGrid(randomForestClassifier.maxDepth, Array(8))
      .addGrid(randomForestClassifier.impurity, Array("gini"))

      .build()

    // define cross validation stage to search through the parameters
    // K-Fold cross validation with BinaryClassificationEvaluator
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    // fit will run cross validation and choose the best set of parameters
    // this will take some time to run
    val cvModel = cv.fit(trainingData)

    // test cross validated model with test data
    val cvPredictionDf = cvModel.transform(testData)
    println("evaluation:")
    println(evaluator.evaluate(cvPredictionDf))
    cvPredictionDf.show(10)

    Evaluator.retrieveMetrics(cvPredictionDf, "results/resultRF.txt")

    cvModel.write.overwrite().save("model/randomForestModel")
    Tools.saveDataFrameToCsv(cvPredictionDf.select($"label", $"prediction"), "RANDOMF")








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

  }*/

}