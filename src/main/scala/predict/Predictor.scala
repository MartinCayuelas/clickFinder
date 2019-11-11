package predict

import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, SparkSession}
import utils.Tools

object Predictor {

  /**
   * Predict click or not click for each lines of a json file
   * @param spark : spark session
   * @param dataPath : path of the json file to analyze
   */
  def predict(spark: SparkSession, dataPath: String, name: String): Unit = {
    import spark.implicits._

    val data = Tools.readDataFrame(dataPath)
    val df = data.drop("label").withColumn("id", monotonically_increasing_id())


    val dataBeforeAssembling = Tools.retrieveDataFrameCleaned(dataPath)

    val featuresCols = Array("appOrSite", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val dataAssembled = assembler.transform(dataBeforeAssembling)


    //val model = LogisticRegressionModel.load("model/randomForestModel").setPredictionCol("label").setFeaturesCol("features")
   val model = RandomForestClassificationModel.load(s"models/${name}")

    val predictions = model
      .transform(dataAssembled)
      .withColumn("label", when($"label" === 0.0, false).otherwise(true))

    val labelColumn = predictions.select("label").withColumn("idl", monotonically_increasing_id())
    val dataFrameToSave = labelColumn.join(df, $"idl" === $"id", "left_outer").drop("id").drop("idl")
    def stringify(c: Column): Column = concat(lit("["), concat_ws(",", c), lit("]"))

    Tools.saveDataFrameToCsv(dataFrameToSave.withColumn("size", stringify(col("size"))), s"output/${name}")
  }

  /**
   * Predict click or not click for each lines of a json file
   * @param spark : spark session
   * @param dataPath : path of the json file to analyze
   */
  def predict1000(spark: SparkSession, dataPath: String): Unit = {

    import spark.implicits._
    println("Start*************************************")
    val now = System.nanoTime

    val data = Tools.limitDataFrame(Tools.readDataFrame(dataPath), 1000)
    println("Numbers of lines : "+data.count())

    val dataBeforeAssembling = Tools.retrieveDataFrameCleaned(dataPath)

    val featuresCols = Array("appOrSite", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val dataAssembled = assembler.transform(dataBeforeAssembling)


    //val model = LogisticRegressionModel.load("model/randomForestModel").setPredictionCol("label").setFeaturesCol("features")
    val model = RandomForestClassificationModel.load("models/randomForestModel")

    val predictions = model
      .transform(dataAssembled)
      .withColumn("label", when($"label" === 0.0, false).otherwise(true))

    val timeElapsed = System.nanoTime - now
    // 1 second = 1_000_000_000 nano seconds// 1 second = 1_000_000_000 nano seconds

    val elapsedTimeInSecond = timeElapsed.asInstanceOf[Double] / 1000000000

    println("Time for 1000 predictions :"+elapsedTimeInSecond + " seconds")
    println("*************************************************")

  }




}