package train
import clean.DataCleaner
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession}
import utils.Tools

object CrossValidation {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()

    val data = Tools.limitDataFrame(Tools.retrieveDataFrameCleaned(),10)
    val featuresCols = Array("appOrSite", "timestamp", "size", "os", "bidFloor", "type", "exchange", "media", "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10", "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19", "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")


    val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 12345)
    // Configure an ML pipeline, which consists of 2 stages: hashingTF, and lr.
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)  // Use 3+ in practice
      .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(training)

    // Make predictions on test documents. cvModel uses the best model found (lrModel).
    val predictions = cvModel.transform(test)
      .select("bidFloor", "label", "probability", "train")
      .collect()
      .foreach { case Row(bidFloor: Double, label: Int, prob: Vector, prediction: Double) =>
        println(s"($bidFloor, $label) --> prob=$prob, prediction=$prediction")
      }

    //print("Model Accuracy: " + evaluatorBis.evaluate(predictions))

    spark.stop()
  }

}
