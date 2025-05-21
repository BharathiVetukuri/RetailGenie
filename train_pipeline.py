from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import AmlCompute
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies


def create_environment(ws):
    """Create and register the Azure ML environment."""
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
        ],
    )
    env.register(workspace=ws)
    return env


def main():
    try:
        # Get workspace
        ws = Workspace.from_config()
        print("Connected to workspace:", ws.name)

        # Get compute target
        compute_target = AmlCompute(ws, "retailgenie-cluster")
        print("Using compute target:", compute_target.name)

        # Create environment
        env = create_environment(ws)
        print("Created environment:", env.name)

        # Create pipeline data for model outputs
        model_output = PipelineData(
            name="model_output", datastore=ws.get_default_datastore()
        )

        # Create run configuration
        run_config = RunConfiguration()
        run_config.environment = env

        # Create training steps
        sql_train_step = PythonScriptStep(
            name="train_sql_generator",
            script_name="train_sqlgen_t5_local.py",
            compute_target=compute_target,
            source_directory="code",
            runconfig=run_config,
            arguments=["--output_dir", model_output.as_mount()],
            allow_reuse=True,
        )

        intent_train_step = PythonScriptStep(
            name="train_intent_classifier",
            script_name="train_intent_classifier_local.py",
            compute_target=compute_target,
            source_directory="code",
            runconfig=run_config,
            arguments=["--output_dir", model_output.as_mount()],
            allow_reuse=True,
        )

        # Create pipeline
        pipeline = Pipeline(workspace=ws, steps=[sql_train_step, intent_train_step])

        # Submit pipeline
        experiment = Experiment(ws, "retailgenie-training")
        run = experiment.submit(pipeline)
        print("Submitted pipeline run:", run.id)

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

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
