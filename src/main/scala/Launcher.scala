import org.apache.spark.sql.SparkSession
import predicting.Predictor
object Launcher extends App {
  override def main(args: Array[String]): Unit = {
    //Create a spark Session
    val spark = SparkSession
      .builder()
      .appName("ClickFinder")
      .config("spark.master", "local")
      .getOrCreate()


    Predictor.predict(spark,"data-10.json")

    spark.close()
  }
}