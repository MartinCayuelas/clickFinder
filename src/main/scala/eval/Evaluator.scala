package eval

import java.io.{BufferedWriter, FileWriter, File}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import utils.Tools

object Evaluator {

  /**
   * Print accuracy, confusion matrix, recall and precision for a prediction DataFrame
   *
   * @param predictions : DataFrame to analyze
   * @param filePath : path of the file for results saving
   */
  def retrieveMetrics(predictions: DataFrame, filePath: String): Unit = {
    val predictionsAndLabels = predictions.select("prediction", "label").rdd
      .map(row => (row.getDouble(0),
        row.get(1).asInstanceOf[Int].toDouble))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"accuracy: ${accuracy}")

    val metrics = new MulticlassMetrics(predictionsAndLabels)
    val confusionMatrix = metrics.confusionMatrix

    println(s" Confusion Matrix:\n ${confusionMatrix.toString}")

    val recallClick = confusionMatrix.apply(1,1) / (confusionMatrix.apply(1,1) + confusionMatrix.apply(0,1))
    val accuracyClick = confusionMatrix.apply(1,1) / (confusionMatrix.apply(1,1) + confusionMatrix.apply(1,0))

    val recallNonClick = confusionMatrix.apply(0,0) / (confusionMatrix.apply(0,0) + confusionMatrix.apply(1,0))
    val accuracyNonClick = confusionMatrix.apply(0,0) / (confusionMatrix.apply(0,0) + confusionMatrix.apply(0,1))

    val fMesureClick = (2 * (accuracyClick * recallClick)) / (accuracyClick + recallClick)
    val fMesureNonClick = (2 * (accuracyNonClick * recallNonClick)) / (accuracyNonClick + recallNonClick)

    println(s"Recall (Clicks): ${recallClick}")
    println(s"Accuracy (Clicks): ${accuracyClick}")
    println(s"F-Mesure (Clicks): ${fMesureClick}")

    println(s"Recall (Non-Clicks): ${recallNonClick}")
    println(s"Accuracy (Non-Clicks): ${accuracyNonClick}")
    println(s"F-Mesure (Non-Clicks): ${fMesureNonClick}")

    Tools.writeFile(s"accuracy: ${accuracy} \n", filePath)
    Tools.writeFile(s"Confusion Matrix:\n ${confusionMatrix.toString}", filePath)
    Tools.writeFile(s"Recall click: ${recallClick}\n", filePath)
    Tools.writeFile(s"Accuracy click: ${accuracyClick}\n", filePath)
    Tools.writeFile(s"Recall non click: ${recallNonClick}\n", filePath)
    Tools.writeFile(s"Accuracy non click: ${accuracyNonClick}\n", filePath)
  }
}
