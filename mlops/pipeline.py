from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential
import yaml
import os


def load_config():
    with open("azure_config.yaml", "r") as file:
        return yaml.safe_load(file)["azure"]


@pipeline
def retailgenie_pipeline(
    training_data: Input, validation_data: Input, model_output: Output
):
    # Training step
    train_step = train_component(
        training_data=training_data,
        validation_data=validation_data,
        model_output=model_output,
    )

    # Evaluation step
    eval_step = evaluate_component(
        model_input=train_step.outputs.model_output, test_data=validation_data
    )

    # Model registration step
    register_step = register_component(
        model_input=train_step.outputs.model_output,
        evaluation_metrics=eval_step.outputs.metrics,
    )

    return {"model": register_step.outputs.model, "metrics": eval_step.outputs.metrics}


def setup_pipeline():
    # Load configuration
    config = load_config()

    # Initialize credentials
    credential = DefaultAzureCredential()

    # Initialize MLClient
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace"]["name"],
    )

    # Create pipeline
    pipeline_job = retailgenie_pipeline(
        training_data=Input(type="uri_file", path="data/train.csv"),
        validation_data=Input(type="uri_file", path="data/val.csv"),
        model_output=Output(type="uri_folder", path="models/"),
    )

    # Set compute target
    pipeline_job.settings.default_compute = config["compute"]["training"]["name"]

    # Submit pipeline
    pipeline_run = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="retailgenie-training"
    )

    print(f"Pipeline submitted. Run ID: {pipeline_run.name}")
    return pipeline_run


if __name__ == "__main__":
    setup_pipeline()
