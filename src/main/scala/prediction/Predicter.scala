import cleaning.DataAnalysis.spark
import cleaning.DataCleaner
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{col, when}


object Predicter {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    val raw_data = DataCleaner.retrieveDataFrame()

    val raw_data_pos = raw_data.filter(col("label") === 1)
    val raw_data_neg = raw_data.filter(col("label") === 0)

    val featuresCols = Array("appOrSite", "timestamp", "size", "os", "bidFloor", "type")
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val data_pos = assembler.transform(raw_data_pos).select( $"features", $"label")
    val data_neg = assembler.transform(raw_data_neg).select( $"features", $"label")

    val Array(trainingDataPos, testDataPos) = data_pos.randomSplit(Array(0.8, 0.2))
    val Array(trainingDataNeg, testDataNeg) = data_neg.randomSplit(Array(0.8, 0.2))

    val trainingData = trainingDataPos.union(trainingDataNeg)
    val testData = testDataPos.union(testDataNeg)

    //TODO - Algorithms predict only 200k lines?

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