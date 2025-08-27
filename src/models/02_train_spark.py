import os
import json

import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# ---------- Spark ----------
spark = (
    SparkSession.builder
    .appName("titanic-train-full-pipeline")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)

# ---------- Data ----------
# Use the processed parquet produced by the preprocess stage
df = spark.read.parquet("data/processed/train.parquet")

# Ensure schema types (cast numerics to double to avoid assembler issues)
num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
for c in num_cols:
    df = df.withColumn(c, col(c).cast("double"))

# These are the categorical columns we’ll encode
cat_cols = ["Sex", "Embarked", "Pclass"]  # keep Pclass as categorical as well

# ---------- Preprocess pipeline ----------
# 1) Impute missing numeric values
imputer = Imputer(
    strategy="median",
    inputCols=num_cols,
    outputCols=[f"{c}_imp" for c in num_cols],
)

# 2) Index categoricals (keep unseen/invalid)
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in cat_cols
]

# 3) One-hot encode categoricals
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_cols],
    outputCols=[f"{c}_oh" for c in cat_cols],
    handleInvalid="keep"
)

# 4) Assemble features (keep rows even if something unexpected appears)
feature_cols = [f"{c}_imp" for c in num_cols] + [f"{c}_oh" for c in cat_cols]
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep"
)

# 5) Classifier
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="Survived",
    seed=42
)

pipeline = Pipeline(stages=[imputer] + indexers + [encoder, assembler, rf])

# Train/valid split
train_df, valid_df = df.randomSplit([0.8, 0.2], seed=42)

# Small hyperparameter grid
grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [50, 100])
    .addGrid(rf.maxDepth, [5, 8])
    .build()
)

evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="Survived",
    metricName="areaUnderROC",
)

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2,
)

# ---------- Fit ----------
cv_model = cv.fit(train_df)

# ---------- Evaluate ----------
val_auc = evaluator.evaluate(cv_model.transform(valid_df))
print(f"Validation AUC: {val_auc:.6f}")

# ---------- Save metrics for DVC ----------
with open("metrics.json", "w") as f:
    json.dump({"val_auc": float(val_auc)}, f)
print("✅ Wrote metrics.json")

# ---------- Save fitted PipelineModel so DVC out 'models/spark_model' is produced ----------
out_path = "models/spark_model"
# overwrite OK so `dvc repro` is idempotent
cv_model.bestModel.write().overwrite().save(out_path)
print(f"✅ Saved fitted PipelineModel to {out_path}")

# ---------- (Optional) Log to MLflow ----------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("titanic-train")

with mlflow.start_run() as run:
    mlflow.log_metric("val_auc", float(val_auc))
    # Log full Spark model (optional but nice)
    mlflow.spark.log_model(
        spark_model=cv_model.bestModel,
        artifact_path="model",
        registered_model_name="titanic_rf"
    )
    print(f"🧪 MLflow run_id: {run.info.run_id}")

spark.stop()
