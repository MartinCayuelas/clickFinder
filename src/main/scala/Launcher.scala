import org.apache.spark.sql.SparkSession
import predict.Predictor

object Launcher extends App {
  override def main(args: Array[String]): Unit = {
    //Create a spark Session
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()

    if(args.length == 0) println("No args given, exiting.")
    else if(args(0) == "predict")  Predictor.predict(spark,args(1))
    else println("")


    spark.close()
  }
}