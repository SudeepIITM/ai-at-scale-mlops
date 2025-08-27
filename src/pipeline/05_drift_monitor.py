import os
import json
import time
import math
import subprocess
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import mlflow

# --------------------------
# Defaults (override via env)
# --------------------------
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT   = os.getenv("DRIFT_EXPERIMENT", "titanic-drift")

# Model info (for tagging / your retrain)
MODEL_NAME   = os.getenv("MODEL_NAME", "titanic_rf")
MODEL_STAGE  = os.getenv("MODEL_STAGE", "Production")

# Files
TRAIN_PATH   = os.getenv("TRAIN_PATH", "data/raw/train.csv")
BASELINE_FP  = os.getenv("BASELINE_JSON", "artifacts/baseline_stats.json")
NEW_DATA_FP  = os.getenv("NEW_DATA", "data/raw/new_batch.csv")

# PSI configuration
NUM_COLS     = os.getenv("NUM_COLS", "Age,SibSp,Parch,Fare").split(",")
N_BINS       = int(os.getenv("PSI_BINS", "10"))
PSI_WARN     = float(os.getenv("PSI_WARN", "0.10"))   # >0.10 moderate drift
PSI_ALERT    = float(os.getenv("PSI_ALERT", "0.25"))  # >0.25 major drift

# Auto-train command (runs if ALERT)
AUTOTRAIN_CMD = os.getenv(
    "AUTOTRAIN_CMD",
    "python src/models/02_train_spark.py && python src/models/03_register.py"
)

# If NEW_DATA is missing and this is "1", create a synthetic batch for testing
NEW_DATA_AUTOGEN = os.getenv("NEW_DATA_AUTOGEN", "0") == "1"


# --------------------------
# Helpers
# --------------------------
def ensure_baseline(train_csv: str, out_json: str, cols: List[str], n_bins: int) -> Dict:
    """Create baseline JSON if missing; otherwise load and validate it."""
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            data = json.load(f)
        # accept both our new schema and a legacy simple schema
        if "num_cols" in data and "bins" in data:
            return data
        # migrate legacy format: {col: {bin_edges: [...]}, ...}
        if all(isinstance(v, dict) and "bin_edges" in v for v in data.values()):
            migrated = {"num_cols": list(data.keys()), "bins": data, "n_bins": n_bins}
            with open(out_json, "w") as f:
                json.dump(migrated, f, indent=2)
            return migrated
        raise ValueError(f"Unexpected baseline schema in {out_json}")

    # Build new baseline from train.csv (equal-frequency edges)
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Missing baseline and training data:\n"
            f"- baseline: {out_json}\n- train: {train_csv}\n"
            f"Provide train.csv or pre-create the baseline JSON."
        )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    df = pd.read_csv(train_csv)

    baseline = {"num_cols": cols, "bins": {}, "n_bins": n_bins}
    for col in cols:
        x = df[col].dropna().values
        if x.size == 0:
            edges = [-float("inf"), float("inf")]
        else:
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.unique(np.quantile(x, qs))
            # make open-ended
            edges[0] = -float("inf")
            edges[-1] = float("inf")
            if len(edges) < 2:
                edges = [-float("inf"), float("inf")]
        baseline["bins"][col] = {"bin_edges": list(map(float, edges))}

    with open(out_json, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"✅ Created baseline at {out_json}")
    return baseline


def autogen_new_batch(train_csv: str, out_csv: str) -> None:
    """Create a synthetic drifted batch if requested."""
    if os.path.exists(out_csv):
        return
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Requested NEW_DATA_AUTOGEN but {train_csv} not found."
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.read_csv(train_csv)
    new = df.sample(n=min(300, len(df)), random_state=123).copy()
    # induce some drift to demonstrate PSI
    if "Age" in new:
        new["Age"] = new["Age"].fillna(new["Age"].median()) + 8.0
    if "Fare" in new:
        new["Fare"] = new["Fare"].fillna(new["Fare"].median()) * 1.5
    new.to_csv(out_csv, index=False)
    print(f"🧪 Created synthetic drift batch at {out_csv}")


def psi_from_counts(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """PSI = sum (pi - qi) * ln(pi/qi) over bins."""
    p = p.astype(float)
    q = q.astype(float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    terms = (p - q) * np.log((p + eps) / (q + eps))
    return float(np.sum(terms))


def bin_counts(x: np.ndarray, edges: List[float]) -> np.ndarray:
    return np.histogram(x, bins=np.array(edges, dtype=float))[0]


def compute_psi(baseline_cfg: Dict, df_new: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    psi_map: Dict[str, float] = {}
    rows = []

    for col in baseline_cfg["num_cols"]:
        edges = baseline_cfg["bins"][col]["bin_edges"]
        xn = df_new[col].dropna().values
        q = bin_counts(xn, edges)
        # baseline used equal-frequency edges → assume uniform baseline per bin
        p = np.ones(len(edges) - 1, dtype=float)
        p /= p.size
        psi_val = psi_from_counts(p, q)
        psi_map[col] = psi_val
        rows.append({"feature": col, "psi": psi_val, "bins": len(edges) - 1, "new_total": int(q.sum())})

    table = pd.DataFrame(rows)
    return psi_map, table


def maybe_autotrain(reason: str) -> int:
    print(f"⚠️ Drift threshold breached ({reason}). Auto-retraining with:\n  {AUTOTRAIN_CMD}")
    try:
        ret = subprocess.run(AUTOTRAIN_CMD, shell=True)
        print(f"Auto-train return code: {ret.returncode}")
        return ret.returncode
    except Exception as e:
        print(f"Auto-train failed: {e}")
        return -1


# --------------------------
# Main
# --------------------------
def main():
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Ensure baseline exists (create if missing)
    baseline_cfg = ensure_baseline(TRAIN_PATH, BASELINE_FP, NUM_COLS, N_BINS)

    # Ensure new data exists (optionally auto-generate)
    if not os.path.exists(NEW_DATA_FP):
        if NEW_DATA_AUTOGEN:
            autogen_new_batch(TRAIN_PATH, NEW_DATA_FP)
        else:
            raise SystemExit(f"Missing NEW_DATA file: {NEW_DATA_FP}\n"
                             f"• Provide your incoming batch CSV, or\n"
                             f"• set NEW_DATA_AUTOGEN=1 to create a synthetic batch from {TRAIN_PATH}")

    df_new = pd.read_csv(NEW_DATA_FP)
    missing = [c for c in baseline_cfg["num_cols"] if c not in df_new.columns]
    if missing:
        raise SystemExit(f"New data missing columns: {missing}")

    # Compute PSI
    psi_map, table = compute_psi(baseline_cfg, df_new)

    # Log to MLflow
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"drift_{int(time.time())}") as run:
        run_id = run.info.run_id
        exp_id = run.info.experiment_id

        # metrics
        for col, val in psi_map.items():
            mlflow.log_metric(f"psi_{col}", float(val))

        worst_feat, worst_val = max(psi_map.items(), key=lambda kv: kv[1])
        mlflow.log_metric("psi_worst", float(worst_val))

        # artifacts
        os.makedirs("artifacts/drift", exist_ok=True)
        report_csv = "artifacts/drift/psi_report.csv"
        table.to_csv(report_csv, index=False)
        mlflow.log_artifact(report_csv)

        # params/tags for context
        mlflow.log_param("psi_bins", N_BINS)
        mlflow.log_param("psi_warn", PSI_WARN)
        mlflow.log_param("psi_alert", PSI_ALERT)

        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("model_stage", MODEL_STAGE)
        mlflow.set_tag("new_data_path", NEW_DATA_FP)
        mlflow.set_tag("baseline_path", BASELINE_FP)
        mlflow.set_tag("psi_worst_feature", worst_feat)

        base = MLFLOW_URI.rstrip("/")
        print(f"🏃 View run {run.data.tags.get('mlflow.runName', '') or run_id} at: {base}/#/experiments/{exp_id}/runs/{run_id}")
        print(f"🧪 View experiment at: {base}/#/experiments/{exp_id}")

    # Status + (optional) retrain
    status = "OK"
    trigger = None
    if any(v >= PSI_ALERT for v in psi_map.values()):
        status = "ALERT"
        trigger = f">= {PSI_ALERT}"
    elif any(v >= PSI_WARN for v in psi_map.values()):
        status = "WARN"
        trigger = f">= {PSI_WARN}"

    print("PSI per feature:", psi_map)
    print(f"Overall drift status: {status}")

    if status == "ALERT":
        rc = maybe_autotrain(trigger)
        print(f"Auto-train finished with code {rc}")


if __name__ == "__main__":
    main()
