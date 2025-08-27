import os, time, json
import mlflow, mlflow.spark
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Optional memory tracking on the driver
try:
    import psutil
    HAVE_PSUTIL = True
except Exception:
    HAVE_PSUTIL = False

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "titanic-train-sweep")
TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

def build_spark():
    # Let spark-submit pass resources; just set a reasonable shuffle default if not provided
    return (
        SparkSession.builder
        .appName("titanic-train-full-pipeline")
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "200"))
        .getOrCreate()
    )

def main():
    t0 = time.time()
    proc = psutil.Process() if HAVE_PSUTIL else None
    rss_start = proc.memory_info().rss if proc else None

    spark = build_spark()
    sc = spark.sparkContext
    conf = sc.getConf()

    # ---------- Data ----------
    df = (spark.read.option("header", True).option("inferSchema", True)
          .csv("data/raw/train.csv"))

    # Basic cleanup
    df = df.withColumn("Sex", when(col("Sex") == "male", "male").otherwise("female"))

    cat_cols = ["Sex", "Embarked", "Pclass"]
    num_cols = ["Age", "SibSp", "Parch", "Fare"]

    # ---------- Preprocess ----------
    imputers = Imputer(strategy="median", inputCols=num_cols, outputCols=[c + "_imp" for c in num_cols])
    indexers = [StringIndexer(handleInvalid="keep", inputCol=c, outputCol=c + "_idx") for c in cat_cols]
    enc = OneHotEncoder(handleInvalid="keep",
                        inputCols=[c + "_idx" for c in cat_cols],
                        outputCols=[c + "_oh" for c in cat_cols])

    feat_cols = [c + "_imp" for c in num_cols] + [c + "_oh" for c in cat_cols]
    vec = VectorAssembler(inputCols=feat_cols, outputCol="features")

    rf = RandomForestClassifier(featuresCol="features", labelCol="Survived", seed=42)

    full_pipeline = Pipeline(stages=[imputers] + indexers + [enc, vec, rf])

    # Split
    train_df, valid_df = df.randomSplit([0.8, 0.2], seed=42)

    # Small grid
    grid = (ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100])
            .addGrid(rf.maxDepth, [5, 8])
            .build())

    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="Survived",
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=full_pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=int(os.getenv("CV_PARALLELISM", "2"))
    )

    # ---------- Train ----------
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log Spark resource settings (what we are sweeping)
        params = {
            "spark.executor.instances": conf.get("spark.executor.instances", "N/A"),
            "spark.executor.cores":     conf.get("spark.executor.cores",     "N/A"),
            "spark.executor.memory":    conf.get("spark.executor.memory",    "N/A"),
            "spark.driver.memory":      conf.get("spark.driver.memory",      "N/A"),
            "spark.sql.shuffle.partitions": conf.get("spark.sql.shuffle.partitions", "N/A"),
            "cv_parallelism": os.getenv("CV_PARALLELISM", "2"),
        }
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = cv.fit(train_df)
        val_auc = evaluator.evaluate(model.transform(valid_df))
        mlflow.log_metric("val_auc", float(val_auc))

        # timing + memory
        runtime_s = time.time() - t0
        mlflow.log_metric("runtime_seconds", runtime_s)

        if HAVE_PSUTIL:
            rss_end = psutil.Process().memory_info().rss
            mlflow.log_metric("driver_rss_start_mb", rss_start / 1e6 if rss_start else 0.0)
            mlflow.log_metric("driver_rss_end_mb",   rss_end / 1e6)
        # Log configured executor memory so you have a memory axis in reports
        mlflow.log_param("cfg_executor_memory_mb",
                         _mem_to_mb(conf.get("spark.executor.memory", "N/A")))

        # Log model (whole pipeline)
        mlflow.spark.log_model(
            spark_model=model.bestModel,
            artifact_path="model",
        )

        print(f"Validation AUC: {val_auc:.4f} | runtime: {runtime_s:.1f}s")
    spark.stop()

def _mem_to_mb(s):
    # Accept "2g", "4096m", "N/A"
    s = str(s).lower()
    try:
        if s.endswith("g"):
            return int(float(s[:-1]) * 1024)
        if s.endswith("m"):
            return int(float(s[:-1]))
        return int(s)
    except Exception:
        return -1

if __name__ == "__main__":
    main()
