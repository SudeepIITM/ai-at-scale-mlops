from fastapi import FastAPI, HTTPException
import os, pandas as pd, mlflow, mlflow.spark
from pyspark.sql import SparkSession

app = FastAPI(title="Titanic API (Spark Pipeline)")
MODEL_NAME, STAGE = "titanic_rf", "Production"

spark = (SparkSession.builder.appName("titanic-api-infer").config("spark.ui.showConsoleProgress","false").getOrCreate())

def load_model():
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))
        m = mlflow.spark.load_model(f"models:/{MODEL_NAME}/{STAGE}")
        print(f"✅ Loaded {MODEL_NAME}@{STAGE}")
        return m
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

model = load_model()

@app.get("/")
def root():
    return {"status":"ok","model_loaded": model is not None}

@app.post("/predict")
def predict(payload: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Ensure MLflow is running and model is in Production.")
    try:
        pdf = pd.DataFrame([payload])           # raw Titanic fields
        sdf = spark.createDataFrame(pdf)        # to Spark DF
        out = model.transform(sdf).select("prediction").toPandas()
        return {"prediction": int(out.loc[0,"prediction"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
