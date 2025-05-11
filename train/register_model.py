# train/register_model.py
import mlflow

# Connect to your MLflow tracking server
mlflow.set_tracking_uri("http://129.114.25.37:8000")
MODEL_NAME = "LLaRA-Rec-Model"

def get_latest_run():
    """
    Return the latest finished run ID from the 'llara-classifier' experiment.
    """
    runs = mlflow.search_runs(
        experiment_names=["llara-classifier"],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"]
    )
    return runs.iloc[0].run_id

# Register the model from the latest run into the MLflow model registry
run_id = get_latest_run()
model_uri = f"runs:/{run_id}/models/llara_model.pt"
mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

print(f"Registered model from run {run_id} as {MODEL_NAME}")
 