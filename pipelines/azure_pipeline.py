from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import AmlCompute
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
import os


def get_workspace():
    """Get or create Azure ML workspace."""
    try:
        ws = Workspace.from_config()
        print("Found workspace:", ws.name)
    except:
        ws = Workspace.create(
            name="retailgenie-workspace",
            subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID"),
            resource_group="retailgenie-rg",
            create_resource_group=True,
            location="eastus",
        )
        print("Created workspace:", ws.name)
    return ws


def get_compute_target(ws):
    """Get or create compute target."""
    compute_name = "retailgenie-cluster"

    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print("Found existing compute target:", compute_name)
            return compute_target

    print("Creating new compute target:", compute_name)
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_NC6",
        min_nodes=1,
        max_nodes=2,
        idle_seconds_before_scaledown=300,
    )

    compute_target = AmlCompute.create(ws, compute_name, provisioning_config)
    compute_target.wait_for_completion(show_output=True)
    return compute_target


def create_environment(ws):
    """Create Azure ML environment."""
    env = Environment("retailgenie-env")
    env.python.conda_dependencies = CondaDependencies.create(
        python_version="3.9",
        pip_packages=[
            "transformers==4.30.2",
            "torch==2.0.1",
            "pandas==2.0.3",
            "numpy==1.24.3",
            "scikit-learn==1.3.0",
            "matplotlib==3.7.2",
            "gradio==3.40.1",
            "azureml-sdk==1.50.0",
            "azure-storage-blob==12.17.0",
            "azure-identity==1.13.0",
            "azure-keyvault-secrets==4.7.0",
            "azure-ml==1.0.0",
            "python-dotenv==1.0.0",
        ],
    )
    env.register(workspace=ws)
    return env


def create_pipeline(ws, compute_target, env):
    """Create Azure ML pipeline."""
    # Create pipeline data for model outputs
    model_output = PipelineData(
        name="model_output", datastore=ws.get_default_datastore()
    )

    # SQL Generator training step
    sql_train_step = PythonScriptStep(
        name="train_sql_generator",
        script_name="train_sqlgen_t5_local.py",
        compute_target=compute_target,
        source_directory="code",
        runconfig=RunConfiguration(),
        arguments=["--output_dir", model_output.as_mount()],
        allow_reuse=True,
    )

    # Intent Classifier training step
    intent_train_step = PythonScriptStep(
        name="train_intent_classifier",
        script_name="train_intent_classifier_local.py",
        compute_target=compute_target,
        source_directory="code",
        runconfig=RunConfiguration(),
        arguments=["--output_dir", model_output.as_mount()],
        allow_reuse=True,
    )

    # Create pipeline
    pipeline = Pipeline(workspace=ws, steps=[sql_train_step, intent_train_step])

    return pipeline


def main():
    """Main function to create and submit pipeline."""
    # Get workspace
    ws = get_workspace()

    # Get compute target
    compute_target = get_compute_target(ws)

    # Create environment
    env = create_environment(ws)

    # Create pipeline
    pipeline = create_pipeline(ws, compute_target, env)

    # Submit pipeline
    experiment = Experiment(ws, "retailgenie-training")
    run = experiment.submit(pipeline)

    # Wait for completion
    run.wait_for_completion(show_output=True)

    # Publish pipeline
    published_pipeline = pipeline.publish(
        name="retailgenie-training-pipeline",
        description="Training pipeline for RetailGenie models",
        version="1.0",
    )

    print(f"Published pipeline: {published_pipeline.name}")
    print(f"Pipeline ID: {published_pipeline.id}")


if __name__ == "__main__":
    main()
