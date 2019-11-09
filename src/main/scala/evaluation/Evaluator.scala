package evaluation

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object Evaluator {

  def printConfusionMatrix(predictions: DataFrame): Unit = {
/*
    val prediction: RDD[(Double, Double)] =
      predictions
      .select("prediction", "label")
      .as[(Double, Double)].rdd

    val metrics = new MulticlassMetrics(prediction)
    println(metrics.confusionMatrix)
    */

  }

  def retrieveMetrics(predictions: DataFrame): Unit = {
    val predictionsAndLabelsN = predictions.select("prediction", "label").rdd
      .map(row => (row.get(0).asInstanceOf[Int].toDouble,
        row.get(1).asInstanceOf[Int].toDouble))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"accuracy: ${accuracy}")

    val metricsN = new MulticlassMetrics(predictionsAndLabelsN)
    val confusionMatrixN = metricsN.confusionMatrix

    // compute the false positive rate per label

    println(s" Confusion Matrix\n ${confusionMatrixN.toString}\n")
    /*
    val nbTrue = pipelineTestingData.filter(df("label")=== 1.0).count().toDouble
    val nbFalse = pipelineTestingData.filter(df("label")=== 0.0).count().toDouble

    println(s"ACCU : ${accuracy*100}")
    println(s"RENTA : ${(confusionMatrixN.apply(1,1)/(confusionMatrixN.apply(1,1)+confusionMatrixN.apply(0,1)))*100}")
    println(s"FINDS : ${(confusionMatrixN.apply(1,1)/nbTrue)*100}")

    println(s" NB FALSE : $nbFalse AND NB TRUE : $nbTrue")
    */
  }
}
