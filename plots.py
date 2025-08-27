import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==== Config ====
MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "titanic-train-automl"  # change if needed
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

# ==== Connect to MLflow ====
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found!")

runs = client.search_runs(exp.experiment_id, "", order_by=["start_time ASC"])

# ==== Collect metrics ====
records = []
for r in runs:
    row = {
        "run_id": r.info.run_id,
        "start_time": r.info.start_time,
    }
    row.update(r.data.metrics)   # pulls all metrics (val_auc, accuracy, etc.)
    records.append(row)

df = pd.DataFrame(records)
print("Available runs:\n", df)

# ==== Plot val_auc across runs ====
if "val_auc" in df.columns:
    plt.figure(figsize=(8,5))
    plt.plot(df["start_time"], df["val_auc"], marker="o", label="val_auc")
    plt.xlabel("Run start time")
    plt.ylabel("Validation AUC")
    plt.title(f"Validation AUC across runs ({EXPERIMENT_NAME})")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "val_auc_runs.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")

# ==== Optional: plot any other metric across runs ====
for metric in ["accuracy", "precision", "recall"]:
    if metric in df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(df["start_time"], df[metric], marker="o", label=metric)
        plt.xlabel("Run start time")
        plt.ylabel(metric)
        plt.title(f"{metric} across runs ({EXPERIMENT_NAME})")
        plt.xticks(rotation=30)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{metric}_runs.png")
        plt.savefig(out_path)
        print(f"Saved {out_path}")
