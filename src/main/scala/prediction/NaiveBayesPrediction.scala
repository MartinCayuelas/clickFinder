package prediction
import cleaning.DataCleaner.spark
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object NaiveBayesPrediction {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  def createModel(filePath: String): NaiveBayesModel = {
    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(spark.sparkContext, filePath)

    // Split data into training (60%) and test (40%).
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println("ACCURACY : " + accuracy)

    model
  }

  def saveModel(model: NaiveBayesModel): Unit = {
    // Save and load model
    model.save(spark.sparkContext, "target/tmp/myNaiveBayesModel")
  }

  def loadModel(modelPath: String): NaiveBayesModel = {
    NaiveBayesModel.load(spark.sparkContext, modelPath)
  }

  def main(args: Array[String]): Unit = {
    createModel("data/fake_data.txt")
    spark.stop()
  }

}