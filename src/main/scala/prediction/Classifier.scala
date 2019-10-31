import cleaning.DataCleaner
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils._


object UnderSampling {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local[4]")
      .appName("example")
      .getOrCreate()

    sparkSession.sparkContext.setLogLevel("ERROR")
    val df = DataCleaner.retrieveDataFrame()
    df.printSchema()

    val amountVectorAssembler = new VectorAssembler().setInputCols(Array("label")).setOutputCol("label_vector")
    val standarScaler = new StandardScaler().setInputCol("label_vector").setOutputCol("label_scaled")
    val dropColumns = Array("appOrSite", "timestamp", "size", "os", "bidFloor", "type")

    val cols = df.columns.filter(column => !dropColumns.contains(column)) ++ Array("label_scaled")
    val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")

    // pipeline 
    val logisticRegression = new LogisticRegression().setLabelCol("label")
    val trainPipeline = new Pipeline().setStages(Array(amountVectorAssembler, standarScaler, vectorAssembler, logisticRegression))

    println("for imbalanced data")
    runPipeline(trainPipeline, df)

    val underSampledDf = underSampleDf(df)
    println("for balanced data")
    val balancedModel = runPipeline(trainPipeline, underSampledDf)

    println("balanced model for full data")
    printScores(balancedModel, df)

  }

  def underSampleDf(df: DataFrame) = {
    val fraudDf = df.filter("label=1")
    val nonFraudDf = df.filter("label=0")
    //random sample the nonFraud to match the value of fraud
    val sampleRatio = fraudDf.count().toDouble / df.count().toDouble
    val nonFraudSampleDf = nonFraudDf.sample(false, sampleRatio)
    fraudDf.union(nonFraudSampleDf)
  }

  def runPipeline(pipeline: Pipeline, df: DataFrame): PipelineModel = {
    val (trainDf, crossDf) = trainTestSplit(df)
    val model = pipeline.fit(trainDf)
    printScores(model, crossDf)
    model
  }

  def printScores(model: PipelineModel, df: DataFrame) = {
    println("test accuracy with pipeline " + accuracyScore(model.transform(df), "label", "prediction"))
    println("test recall for 1 is " + recall(model.transform(df), "label", "prediction", 1))
  }


  def accuracyScore(df: DataFrame, label: String, predictCol: String) = {
    val rdd = df.select(predictCol, label).rdd.map(row ⇒ (row.getDouble(0), row.getInt(1).toDouble))
    new MulticlassMetrics(rdd).accuracy
  }

  def recall(df: DataFrame, labelCol: String, predictCol: String, labelValue: Double) = {
    val rdd = df.select(predictCol, labelCol).rdd.map(row ⇒ (row.getDouble(0), row.getInt(1).toDouble))
    new MulticlassMetrics(rdd).recall(labelValue)
  }

  def trainTestSplit(df: DataFrame, testSize: Double = 0.3): (DataFrame, DataFrame) = {
    val dfs = df.randomSplit(Array(1 - testSize, testSize))
    val trainDf = dfs(0)
    val crossDf = dfs(1)
    (trainDf, crossDf)
  }
}