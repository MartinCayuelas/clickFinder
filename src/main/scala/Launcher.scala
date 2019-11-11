import org.apache.spark.sql.SparkSession
import predict.Predictor
import train.randomforest.Trainer

object Launcher extends App {
  override def main(args: Array[String]): Unit = {
    //Create a spark Session
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()

    if(args.length == 0) println("You should provide an action and a path to your dataset.")
    else if(args(0) == "predict")  Predictor.predict(spark,args(1))
    else if(args(0) == "train")  Trainer.train(spark, args(1))

    spark.close()
  }
}