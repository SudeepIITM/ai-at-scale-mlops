import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
from pyspark.ml import Pipeline
import mlflow

# Paths
input_path = "data/raw/train.csv"
output_path = "data/processed/train.parquet"

spark = (SparkSession.builder
         .appName("titanic-preprocess")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())

mlflow.set_experiment("titanic-preprocess")

with mlflow.start_run():
    # Read Titanic CSV
    df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

    # Fix Sex column
    df = df.withColumn("Sex", when(col("Sex") == "male", "male").otherwise("female"))

    # Feature groups
    cat_cols = ["Sex", "Embarked", "Pclass"]
    num_cols = ["Age", "SibSp", "Parch", "Fare"]

    # Impute numerics
    imputers = Imputer(strategy="median", inputCols=num_cols, outputCols=[c+"_imp" for c in num_cols])

    # Encode categoricals
    indexers = [StringIndexer(handleInvalid="keep", inputCol=c, outputCol=c+"_idx") for c in cat_cols]
    enc = OneHotEncoder(handleInvalid="keep",
                        inputCols=[c+"_idx" for c in cat_cols],
                        outputCols=[c+"_oh" for c in cat_cols])

    # Assemble features
    feat_cols = [c+"_imp" for c in num_cols] + [c+"_oh" for c in cat_cols]
    vec = VectorAssembler(inputCols=feat_cols, outputCol="features")

    # Label
    label_indexer = StringIndexer(inputCol="Survived", outputCol="label")

    pipe = Pipeline(stages=[imputers] + indexers + [enc, vec, label_indexer])
    model = pipe.fit(df)
    out = model.transform(df).select("features", "label")

    # Save output
    out.write.mode("overwrite").parquet(output_path)

    # Log
    mlflow.log_artifact("src/pipeline/01_preprocess_spark.py")
    mlflow.log_param("rows", df.count())

spark.stop()
