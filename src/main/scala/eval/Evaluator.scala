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
    val predictionsAndLabelsN = predictions.select("prediction", "label").rdd
      .map(row => (row.getDouble(0),
        row.get(1).asInstanceOf[Int].toDouble))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"accuracy: ${accuracy}")

    val metricsN = new MulticlassMetrics(predictionsAndLabelsN)
    val confusionMatrixN = metricsN.confusionMatrix
    val confusionMatrix = inverseMatrix(confusionMatrixN)
    println(s" Confusion Matrix:\n ${confusionMatrix(0)(0)}\t${confusionMatrix(0)(1)}\n${confusionMatrix(1)(0)}\t${confusionMatrix(1)(1)}")

    val recallMatrix = confusionMatrix(0)(0) / (confusionMatrix(0)(0) + confusionMatrix(1)(0))
    val accuracyMatrix = confusionMatrix(0)(0) / (confusionMatrix(0)(0) + confusionMatrix(0)(1))

    println(s"Recall: ${recallMatrix}")
    println(s"Accuracy: ${accuracyMatrix}")

    Tools.writeFile(s"accuracy: ${accuracy} \n", filePath)
    Tools.writeFile(s" Confusion Matrix:\n ${confusionMatrix(0)(0)}\t${confusionMatrix(0)(1)}\n${confusionMatrix(1)(0)}\t${confusionMatrix(1)(1)}\n", filePath)
    Tools.writeFile(s"Recall: ${recallMatrix}\n", filePath)
    Tools.writeFile(s"Accuracy: ${accuracyMatrix}\n", filePath)

  }

  def inverseMatrix(matrix: Matrix): Array[Array[Double]] = {
    val trueNegative = matrix.apply(0,0)
    val falsePositive = matrix.apply(0,1)
    val falseNegative = matrix.apply(1,0)
    val truePositive = matrix.apply(1,1)
    val result = Array.ofDim[Double](2,2)

    result(0)(0) = truePositive
    result(1)(1) = trueNegative
    result(1)(0) = falsePositive
    result(0)(1) = falseNegative
    result
  }


}
