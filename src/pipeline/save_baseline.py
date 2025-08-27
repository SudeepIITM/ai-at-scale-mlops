import os, json, numpy as np, pandas as pd

TRAIN_PATH = "data/raw/train.csv"
OUT_PATH   = "artifacts/baseline_stats.json"
NUM_COLS   = ["Age", "SibSp", "Parch", "Fare"]
N_BINS     = 10

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.read_csv(TRAIN_PATH)

baseline = {"num_cols": NUM_COLS, "bins": {}, "n_bins": N_BINS}
for col in NUM_COLS:
    x = df[col].dropna().values
    if x.size == 0:
        edges = [-float("inf"), float("inf")]
    else:
        # equal-frequency bin edges; unique to avoid duplicates
        qs = np.linspace(0, 1, N_BINS + 1)
        edges = np.unique(np.quantile(x, qs))
        # make open-ended at extremes
        edges[0]  = -float("inf")
        edges[-1] =  float("inf")
        if len(edges) < 2:
            edges = [-float("inf"), float("inf")]
    baseline["bins"][col] = {"bin_edges": list(map(float, edges))}

with open(OUT_PATH, "w") as f:
    json.dump(baseline, f, indent=2)

print(f"✅ Wrote baseline to {OUT_PATH}")
