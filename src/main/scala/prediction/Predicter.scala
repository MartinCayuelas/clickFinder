import cleaning.DataCleaner
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler


object Predicter {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    val raw_data = DataCleaner.retrieveDataFrame()

    val featuresCols = Array("appOrSite", "timestamp", "size", "os", "bidFloor", "type")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")
    val data = assembler.transform(raw_data).select( $"features", $"label")

    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

    val logisticRegression = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)
      .setFamily("binomial")
      .setThreshold(0.5)

    val randomForest = new RandomForestClassifier()

    // Fit the model for Logistic Regression
    val logisticRegressionModel = logisticRegression.fit(trainingData)
    val randomForestModel = randomForest.fit(trainingData)

    // Get predictions
    val predictions = logisticRegressionModel.transform(testData)
    val predictionsForest = randomForestModel.transform(testData)

    //Evaluator for the classification
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predictions)

    println("accuracy : " + accuracy)
    println(s"Coefficients: ${logisticRegressionModel.coefficients} Intercept: ${logisticRegressionModel.intercept}")

    DataCleaner.saveDataFrameToCsv(predictions.select($"label", $"prediction"), "predictionLR")
    DataCleaner.saveDataFrameToCsv(predictionsForest.select($"label", $"prediction"), "predictionRF")

    spark.stop()
  }
}