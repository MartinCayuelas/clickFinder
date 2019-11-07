import cleaning.DataCleaner
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._


object Predicter {

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

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    val raw_data = DataCleaner.retrieveDataFrame()
    val featuresCols = Array("appOrSite", "timestamp", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val data = assembler.transform(raw_data).select( $"features", $"label")
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

    /*
    val temp = trainingData.groupBy("label").count()
    val new_train = trainingData.join(temp, "label")
    trainingData)
    val num_labels = trainingData.select(countDistinct(trainingData.score)).first()(0)
    train1 = new_train.withColumn("weight",(new_train.count()/(num_labels * new_train("count"))))
    */

    val balanced_dataset = balanceDataset(trainingData)
    val lr = new LogisticRegression()
      .setWeightCol("classWeightCol")
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setFamily("binomial")

    val logisticRegression = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)
      .setFamily("binomial")
      .setThreshold(0.5)

    val randomForest = new RandomForestClassifier()

    // Fit the model for Logistic Regression
    val logisticRegressionModel = logisticRegression.fit(trainingData)
    val randomForestModel = randomForest.fit(trainingData)
    val balancedLR = lr.fit(balanced_dataset)

    // Get predictions
    val predictions = logisticRegressionModel.transform(testData)
    val predictionsForest = randomForestModel.transform(testData)
    val predictionsBalancedLR = balancedLR.transform(testData)

    //Evaluator for the classification
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predictions)
    val accuracyBLR = evaluator.evaluate(predictionsBalancedLR)

    println("accuracy : " + accuracy)
    println("accuracy (balanced): " + accuracyBLR)
    println(s"Coefficients: ${logisticRegressionModel.coefficients} Intercept: ${logisticRegressionModel.intercept}")

    DataCleaner.saveDataFrameToCsv(predictions.select($"label", $"prediction"), "predictionLR")
    DataCleaner.saveDataFrameToCsv(predictionsForest.select($"label", $"prediction"), "predictionRF")
    DataCleaner.saveDataFrameToCsv(predictionsBalancedLR.select($"label", $"prediction"), "predictionBLR")

    spark.stop()
  }
}