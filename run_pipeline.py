import os
from google.cloud import aiplatform

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Configuration
PROJECT_ID = "retailgenie-459802"
REGION = "us-central1"
PIPELINE_NAME = "retailgenie-vertex-pipeline"

# Path to the pipeline template
pipeline_template_path = os.path.join(PROJECT_ROOT, "vertex_pipeline.yaml")

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Launch the pipeline
pipeline_job = aiplatform.PipelineJob(
    display_name=PIPELINE_NAME,
    template_path=pipeline_template_path,
    parameter_values={
        "dataset_path": "gs://retailgenie-data/retail_dataset.csv",
        "output_dir": "gs://retailgenie-data/model-outputs",
    },
    enable_caching=False,
)

pipeline_job.run(sync=True)
