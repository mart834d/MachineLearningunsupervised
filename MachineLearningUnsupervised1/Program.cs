using System;
using System.IO;
using MachineLearningUnsupervised1.Model;
using Microsoft.ML;

namespace MachineLearningUnsupervised1
{
    class Program
    {
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "iris.csv");
        static void Main(string[] args)
        {
            var mlContext = new MLContext();


            //import data
            var trainingData = mlContext.Data.LoadFromTextFile<IrisData>(
            path: dataPath,
            hasHeader: false,
            separatorChar: ',');

            //data preparation and training pipeline
            var pipeline = mlContext.Transforms
                .Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));


            //train
            var model = pipeline.Fit(trainingData);

            //prediction
            var dataToMakePredictionOn = new IrisData() { PetalLength = 0.2f, PetalWidth = 5.6f, SepalLength = 1f, SepalWidth = 1.6f };
            var label = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model).Predict(dataToMakePredictionOn);

            Console.WriteLine($"{string.Join(" ", label.Distances)}");
        }
    }
}
