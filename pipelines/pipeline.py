from kfp import dsl
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

@dsl.pipeline(
    name="retailgenie-vertex-pipeline",
    description="Train SQLGen + IntentClassifier + Deploy both models"
)
def retailgenie_pipeline(
    project: str,
    location: str,
    dataset_path: str,
    output_dir: str,
    sql_model_path: str = "gs://retailgenie-data/model-outputs/sqlgen",
    intent_model_path: str = "gs://retailgenie-data/model-outputs/intent"
):

    sqlgen_job = CustomTrainingJobOp(
        display_name="train-sql-generator",
        project=project,
        location=location,
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/retailgenie-459802/retailgenie/retailgenie-train-sqlgen:latest",
                "args": [
                    "--dataset_path", dataset_path,
                    "--output_dir", output_dir
                ]
            }
        }]
    )

    intent_job = CustomTrainingJobOp(
        display_name="train-intent-classifier",
        project=project,
        location=location,
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/retailgenie-459802/retailgenie/retailgenie-train-intent:latest",
                "args": [
                    "--dataset_path", dataset_path,
                    "--output_dir", output_dir
                ]
            }
        }]
    )
