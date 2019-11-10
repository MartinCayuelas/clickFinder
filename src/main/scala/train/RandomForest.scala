package train

import clean.DataCleaner
import eval.Evaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import utils.Tools

object RandomForest {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    val data = Tools.retrieveDataFrameCleaned()
    val raw_data = Tools.limitDataFrame(data, 100000)


    val featuresCols = Array("appOrSite", "timestamp", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    //val data = assembler.transform(raw_data)
    //val balanced_dataset = balanceDataset(data)

    val Array(trainingData, testData) = raw_data.randomSplit(Array(0.8, 0.2), seed = 42)

    val randomForestClassifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(3)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setSeed(42)

    //val randomForestModel = randomForestClassifier.fit(trainingData)

    //val predictionDf = randomForestModel.transform(testData)


    // evaluate model with area under ROC
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")

    // measure the accuracy
   // val accuracy = evaluator.evaluate(predictionDf)
   // println(accuracy)




    // VectorAssembler and StringIndexer are transformers
    // LogisticRegression is the estimator
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

    spark.stop()
  }
}
