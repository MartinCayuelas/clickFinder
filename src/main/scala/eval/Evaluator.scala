package eval

import java.io.{BufferedWriter, FileWriter, File}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object Evaluator {

  /**
   * Print accuracy, confusion matrix, recall and precision for a prediction DataFrame
   *
   * @param predictions : DataFrame to analyze
   */
  def retrieveMetrics(predictions: DataFrame): Unit = {
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

    writeFile(s"accuracy: ${accuracy} \n")
    writeFile(s" Confusion Matrix:\n ${confusionMatrix(0)(0)}\t${confusionMatrix(0)(1)}\n${confusionMatrix(1)(0)}\t${confusionMatrix(1)(1)}\n")
    writeFile(s"Recall: ${recallMatrix}\n")
    writeFile(s"Accuracy: ${accuracyMatrix}\n")

    /*
    val nbTrue = pipelineTestingData.filter(df("label")=== 1.0).count().toDouble
    val nbFalse = pipelineTestingData.filter(df("label")=== 0.0).count().toDouble

    println(s"ACCU : ${accuracy*100}")
    println(s"RENTA : ${(confusionMatrixN.apply(1,1)/(confusionMatrixN.apply(1,1)+confusionMatrixN.apply(0,1)))*100}")
    println(s"FINDS : ${(confusionMatrixN.apply(1,1)/nbTrue)*100}")

    println(s" NB FALSE : $nbFalse AND NB TRUE : $nbTrue")
    */
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

  def writeFile(s: String): Unit = {
    val file = new File("resultLR.txt")
    val bw = new BufferedWriter(new FileWriter(file, true))
    bw.write(s)
    bw.write("\n")
    bw.close()
  }
}
