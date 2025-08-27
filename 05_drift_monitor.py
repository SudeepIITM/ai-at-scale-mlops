import pandas as pd
import numpy as np
from scipy.stats import entropy
import json
import os
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Configuration
EXPERIMENT = "titanic-drift"
BASELINE_PATH = "artifacts/baseline_stats.json"
DRIFT_THRESHOLD = 0.25  # PSI threshold for alerting

def load_baseline():
    """Load baseline statistics"""
    with open(BASELINE_PATH, 'r') as f:
        return json.load(f)

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)"""
    # Create buckets based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[-1] += 1e-6  # Ensure last bucket includes max value
    
    # Calculate expected distribution
    expected_hist, _ = np.histogram(expected, bins=breakpoints)
    expected_perc = expected_hist / len(expected)
    expected_perc[expected_perc == 0] = 1e-6  # Avoid division by zero
    
    # Calculate actual distribution
    actual_hist, _ = np.histogram(actual, bins=breakpoints)
    actual_perc = actual_hist / len(actual)
    actual_perc[actual_perc == 0] = 1e-6  # Avoid division by zero
    
    # Calculate PSI
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return psi

def monitor_drift():
    """Main drift monitoring function"""
    print("🔍 Starting drift monitoring...")
    
    # Load baseline statistics
    baseline = load_baseline()
    print(f"📊 Loaded baseline with features: {list(baseline['distributions'].keys())}")
    
    # Load new data (simulate with existing data for demo)
    # In production, this would be data/raw/new_batch.csv
    try:
        new_data = pd.read_csv("data/raw/train.csv")
        print(f"📈 Loaded new data with {len(new_data)} samples")
    except FileNotFoundError:
        print("❌ New data file not found. Using sample data for demonstration.")
        # For demo purposes, create some drifted data
        new_data = pd.read_csv("data/raw/train.csv")
        # Simulate drift: make passengers older and fares higher
        new_data['Age'] = new_data['Age'] * 1.3  # 30% older
        new_data['Fare'] = new_data['Fare'] * 1.5  # 50% higher fares
    
    # Calculate PSI for each feature
    psi_results = {}
    for feature, baseline_dist in baseline['distributions'].items():
        if feature in new_data.columns and not new_data[feature].isnull().all():
            # Get current data (remove nulls)
            current_data = new_data[feature].dropna()
            if len(current_data) > 0:
                psi = calculate_psi(np.array(baseline_dist), current_data.values)
                psi_results[feature] = psi
                print(f"   {feature}: PSI = {psi:.4f}")
            else:
                print(f"   {feature}: No data available")
        else:
            print(f"   {feature}: Column not found in new data")
    
    return psi_results

def main():
    """Main function"""
    print("=" * 60)
    print("🌊 TITANIC DATA DRIFT MONITORING")
    print("=" * 60)
    
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT)
    
    with mlflow.start_run(run_name=f"drift_{int(pd.Timestamp.now().timestamp())}"):
        # Monitor for drift
        psi_results = monitor_drift()
        
        # Log PSI results
        for feature, psi in psi_results.items():
            mlflow.log_metric(f"psi_{feature}", psi)
        
        # Determine overall drift status
        max_psi = max(psi_results.values()) if psi_results else 0
        mlflow.log_metric("max_psi", max_psi)
        
        print(f"\n📊 PSI per feature: {psi_results}")
        print(f"📈 Maximum PSI: {max_psi:.4f}")
        print(f"🎯 Drift threshold: {DRIFT_THRESHOLD}")
        
        # Check for drift
        if max_psi >= DRIFT_THRESHOLD:
            drift_status = "ALERT"
            print("🚨 Overall drift status: ALERT")
            print(f"⚠️ Drift threshold breached (>= {DRIFT_THRESHOLD}). Auto-retraining...")
            
            # Trigger auto-retraining - UPDATED TO USE SPARK SCRIPT
            retcode = os.system("python src/models/retrain_spark.py")
            print(f"Auto-train return code: {retcode}")
            
            # Log alert
            mlflow.log_param("drift_status", "ALERT")
            mlflow.log_metric("retraining_triggered", 1)
            mlflow.log_dict(psi_results, "psi_results.json")
            
        else:
            drift_status = "OK"
            print("✅ Overall drift status: OK")
            print("   No significant drift detected")
            mlflow.log_param("drift_status", "OK")
            mlflow.log_metric("retraining_triggered", 0)
        
        print("=" * 60)
        print("📋 DRIFT MONITORING COMPLETE")
        print("=" * 60)

if __name__ == "__main__":
    main()
