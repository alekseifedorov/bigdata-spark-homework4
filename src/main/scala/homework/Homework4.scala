package homework


import org.apache.spark._
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy

import scala.io.Source

object Homework4 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Homework4").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val objectsFile = Source.fromFile("Objects.csv").getLines()
    val objects = objectsFile.map(label => {
      label.replace(",", ".").replace("NaN", "0").split(";").map(_.toDouble)
    }
    )

    val targets = Source.fromFile("Target.csv").getLines().map(line => line.toDouble)
    var zippedObjects = for ((label, features) <- (targets zip objects)) yield new LabeledPoint(label, Vectors.dense(features))
    val splitData = sc.parallelize(zippedObjects.toSeq).randomSplit(Array(0.6, 0.4), seed = 123)
    val trainingData = splitData(0)
    val testData = splitData(1)

    // replace negative numbers from data fot the naive Bayes metrics
    val nbTargets = Source.fromFile("Target.csv").getLines().map(line => line.toDouble).map(d => if (d < 0) 0.0 else d)
    var zippedNbObjects = for ((label, features) <- (nbTargets zip objects)) yield new LabeledPoint(label, Vectors.dense(features))
    val splitNbData = sc.parallelize(zippedNbObjects.toSeq).randomSplit(Array(0.6, 0.4), seed = 123)
    val nbTrainingData = splitNbData(0)
    val testNbData = splitNbData(1)


    // SVM and LR models
    val numIterations = 100
    val lrModel = LogisticRegressionWithSGD.train(trainingData, numIterations)
    val svmModel = SVMWithSGD.train(trainingData, numIterations)
    val metrics = Seq(lrModel, svmModel).map { model =>
      val scoreAndLabels = testData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderROC)
    }
    metrics.foreach { case (m, roc) =>
      println(f"$m,  Area under ROC: ${roc * 100.0}%2.4f%%")
    }

    // Naive Bayes model
    val nbModel = NaiveBayes.train(nbTrainingData)
    val nbMetrics = Seq(nbModel).map { model =>
      val scoreAndLabels = testNbData.map { point =>
        val score = nbModel.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderROC)
    }
    nbMetrics.foreach { case (m, roc) =>
      println(f"$m,  Area under ROC: ${roc * 100.0}%2.4f%%")
    }


    // train a logistic regression model on the scaled data, and compute metrics
    val vectors = trainingData.map(lp => lp.features)
    import org.apache.spark.mllib.feature.StandardScaler
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledTrainingData = trainingData.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    val scaledTestData = trainingData.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

    val lrModelScaled = LogisticRegressionWithSGD.train(scaledTrainingData, numIterations)
    val lrPredictionsVsTrue = scaledTestData.map { point =>
      (lrModelScaled.predict(point.features), point.label)
    }

    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
    val lrPr = lrMetricsScaled.areaUnderPR
    val lrRoc = lrMetricsScaled.areaUnderROC
    println(f"${lrModelScaled.getClass.getSimpleName} on scaled data, Area under ROC: ${lrRoc * 100.0}%2.4f%%")

    Seq(1, 2, 3, 4, 5, 10, 20).map(maxTreeDepth => {
      val dtModel = DecisionTree.train(trainingData, Algo.Classification, Entropy, maxTreeDepth)
      val dtMetrics = Seq(dtModel).map { model =>
        val scoreAndLabels = testData.map { point =>
          val score = model.predict(point.features)
          (if (score > 0.5) 1.0 else 0.0, point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderROC)
      }

      dtMetrics.foreach { case (m, roc) =>
        println(f"$m, maxTreeDepth: $maxTreeDepth, Area under ROC: ${roc * 100.0}%2.4f%%")
      }
    })

    sc.stop
  }


}
