import mlflow
from mlflow.tracking import MlflowClient

# Point to your local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Find the "titanic-train" experiment
exp = client.get_experiment_by_name("titanic-train")
if exp is None:
    raise ValueError("Experiment 'titanic-train' not found. Did you run training?")

# Get the best run by validation AUC
runs = client.search_runs(
    [exp.experiment_id],
    order_by=["metrics.val_auc DESC"],
    max_results=1,
)
if not runs:
    raise ValueError("No runs found in experiment 'titanic-train'.")

best_run = runs[0]
print("Best run ID:", best_run.info.run_id)
print("Best val_auc:", best_run.data.metrics.get("val_auc"))

# Model registry name
model_name = "titanic_rf"

# Ensure the registered model exists
try:
    client.create_registered_model(model_name)
    print(f"Created new registered model: {model_name}")
except Exception:
    print(f"Model {model_name} already exists, skipping creation.")

# Register the model artifact
# NOTE: adjust artifact path if your training script logs differently
model_uri = f"runs:/{best_run.info.run_id}/model"
mv = mlflow.register_model(model_uri, model_name)
print(f"Registered model version {mv.version}")

# Promote it to Production
client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"Model {model_name} v{mv.version} is now in Production.")
