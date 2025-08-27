import os, pandas as pd, mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME, STAGE = "titanic_automl", "Production"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")

df = pd.read_csv("data/raw/test.csv")
pred = model.predict(df)
out = df.copy(); out["prediction"] = pred
out.to_csv("predictions_test.csv", index=False)

with mlflow.start_run(run_name=f"predictions_{MODEL_NAME}_{STAGE}"):
    mlflow.log_artifact("predictions_test.csv")
