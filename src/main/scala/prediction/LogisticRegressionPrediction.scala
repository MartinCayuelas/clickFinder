package prediction

import cleaning.DataCleaner.spark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler

object LogisticRegressionPrediction {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  def createModel(df: DataFrame): LogisticRegressionModel = {

    // Transform the list of indexed columns into a single vector column.
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("V1", "V2", "V3","V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11","V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19","V20", "V21", "V22", "V23", "V24"))
      .setOutputCol("features")

    //Create a dataframe with 2 columns: label and features
    val df3 = assembler.transform(df)

    // Split data into training (60%) and test (40%).
    val Array(training, test) = df3.randomSplit(Array(0.7, 0.3))

    val lr = new LogisticRegression()
      .setLabelCol("Class")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    lrModel
  }

  def main(args: Array[String]): Unit = {
    val df: DataFrame = spark.read.format("csv")
      .option("sep", ";")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("data/sonar.csv")
    println(df.printSchema())
    createModel(df)
    spark.stop()
  }

}



// Load training data
//val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")






