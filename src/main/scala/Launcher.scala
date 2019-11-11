import org.apache.spark.sql.SparkSession
import predict.Predictor
import train.randomforest.Trainer

object Launcher extends App {
  override def main(args: Array[String]): Unit = {
    // Create a spark Session

    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()

    if  (args.length < 1) {
      print("Use command train | predict | predict1000.");
    }

    val command = args(0)

    command match {
      case "train" => {
        args match {
          case Array(c, filepath, name) => Trainer.train(spark, filepath, name)
          case _ => println("See Usage")
        }
      }
      case "predict" =>
        args match {
          case Array(c, filepath, name) => Predictor.predict(spark,filepath, name)
          case Array(c, filepath) => Predictor.predict(spark,filepath, "randomForestModel")
          case _ => println("See Usage")
      }
      case "predict1000" => {
        args match {
          case Array(c, filepath) => Predictor.predict1000(spark, filepath)
          case _ => print("See usage")
        }
      }
      case _ => print("See usage.")
    }


    spark.close()
  }
}