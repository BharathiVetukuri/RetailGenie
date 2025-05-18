import os

os.environ["GOOGLE_AUTH_USE_CLIENT_CERTIFICATE"] = "true"

from google.auth import default
from google.cloud import aiplatform

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

credentials, _ = default()

# Initialize Vertex AI using your ADC credentials
aiplatform.init(
    project="retailgenie-459802", location="us-central1", credentials=credentials
)

# Path to the pipeline template
pipeline_template_path = os.path.join(
    PROJECT_ROOT, "pipelines", "retailgenie_pipeline.json"
)

# Submit the pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name="retailgenie-pipeline-v1",
    template_path=pipeline_template_path,
    parameter_values={
        "project": "retailgenie-459802",
        "location": "us-central1",
        "dataset_path": "gs://retailgenie-data/retail_dataset.csv",
        "output_dir": "gs://retailgenie-data/model-outputs",
        "sql_model_path": "gs://retailgenie-data/model-outputs/sqlgen",
        "intent_model_path": "gs://retailgenie-data/model-outputs/intent",
    },
)

pipeline_job.run(sync=True)
