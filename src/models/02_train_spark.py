import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("TitanicTraining").getOrCreate()

# Load Titanic dataset
data = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)

# Basic preprocessing
indexer = StringIndexer(inputCol="Sex", outputCol="SexIndexed")
assembler = VectorAssembler(
    inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "SexIndexed"],
    outputCol="features"
)
rf = RandomForestClassifier(labelCol="Survived", featuresCol="features")

pipeline = Pipeline(stages=[indexer, assembler, rf])

train, test = data.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)

preds = model.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
auc = evaluator.evaluate(preds)
print("Validation AUC:", auc)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("titanic-train")

with mlflow.start_run():
    mlflow.log_metric("val_auc", auc)
    mlflow.spark.log_model(model, "model", registered_model_name="titanic_rf")

spark.stop()
