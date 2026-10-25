"""Log the v2 bundle to MLflow, tagged as production."""
import json
import mlflow

info = json.load(open("bundles/v2/bundle.json"))
mlflow.set_experiment("bracket-stress-bundles")
with mlflow.start_run(run_name="bundle-v2"):
    mlflow.log_params(info["architecture"])
    mlflow.set_tags({
        "stage": "production",
        "training_repo": info["lineage"]["code"]["repo"],
        "training_commit": info["lineage"]["code"]["git_commit"],
        "load_cases": ",".join(info["load_cases"]),
    })
    mlflow.log_artifacts("bundles/v2", artifact_path="bundle")
print("logged bundle-v2")