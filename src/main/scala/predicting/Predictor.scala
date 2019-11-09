package predicting

import cleaning.DataCleaner
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions._

object Predictor {

  def predict(spark: SparkSession, dataPath: String): Unit = {
    import spark.implicits._

    val data = DataCleaner.readDataFrame(dataPath)
    val df = data.drop("label").withColumn("id", monotonically_increasing_id())

    val dataSelected = DataCleaner.selectData(data)
    val dataAssembled = DataCleaner.clean(dataSelected)
    val featuresCols = Array("appOrSite", "timestamp", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

    /*
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")

    val dataAssembled = assembler.transform(dataCleaned)
    */

    //val model = LogisticRegressionModel.load("model/randomForestModel").setPredictionCol("label").setFeaturesCol("features")
   val model = CrossValidatorModel.load("model/randomForestModel")

    val predictions = model
      .transform(dataAssembled)
      .withColumn("label", when($"label" === 0.0, false).otherwise(true))

    val labelColumn = predictions.select("label").withColumn("idl", monotonically_increasing_id())
    val dataFrameToSave = labelColumn.join(df, $"idl" === $"id", "left_outer").drop("id").drop("idl")
    def stringify(c: Column): Column = concat(lit("["), concat_ws(",", c), lit("]"))

    DataCleaner.saveDataFrameToCsv(dataFrameToSave.withColumn("size", stringify(col("size"))), "prediction")
  }
}