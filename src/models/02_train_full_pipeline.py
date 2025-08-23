import os, mlflow, mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = (SparkSession.builder.appName("titanic-train-full").config("spark.ui.showConsoleProgress","false").getOrCreate())

df = spark.read.option("header",True).option("inferSchema",True).csv("data/raw/train.csv")
df = df.withColumn("Sex", when(col("Sex")=="male","male").otherwise("female"))

cat = ["Sex","Embarked","Pclass"]
num = ["Age","SibSp","Parch","Fare"]

imputer = Imputer(strategy="median", inputCols=num, outputCols=[c+"_imp" for c in num])
indexers = [StringIndexer(handleInvalid="keep", inputCol=c, outputCol=c+"_idx") for c in cat]
enc = OneHotEncoder(handleInvalid="keep", inputCols=[c+"_idx" for c in cat], outputCols=[c+"_oh" for c in cat])
vec = VectorAssembler(inputCols=[c+"_imp" for c in num] + [c+"_oh" for c in cat], outputCol="features")
rf = RandomForestClassifier(featuresCol="features", labelCol="Survived", seed=42)

pipe = Pipeline(stages=[imputer]+indexers+[enc, vec, rf])

train, valid = df.randomSplit([0.8,0.2], seed=42)
grid = (ParamGridBuilder().addGrid(rf.numTrees,[50,100]).addGrid(rf.maxDepth,[5,8]).build())
evalr = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
cv = CrossValidator(estimator=pipe, estimatorParamMaps=grid, evaluator=evalr, numFolds=3, parallelism=2)

cvModel = cv.fit(train)
auc = evalr.evaluate(cvModel.transform(valid))
print(f"Validation AUC: {auc:.4f}")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))
mlflow.set_experiment("titanic-train")
with mlflow.start_run() as run:
    mlflow.log_metric("val_auc", auc)
    # LOG THE **FITTED PIPELINEMODEL**
    mlflow.spark.log_model(cvModel.bestModel, "model", registered_model_name="titanic_rf")
    run_id = run.info.run_id
    print("Logged run:", run_id)

client = MlflowClient()
vers = client.search_model_versions("name='titanic_rf'")
latest = max(int(v.version) for v in vers)
client.transition_model_version_stage(name="titanic_rf", version=str(latest), stage="Production", archive_existing_versions=False)
print(f"Promoted titanic_rf v{latest} -> Production")

spark.stop()
