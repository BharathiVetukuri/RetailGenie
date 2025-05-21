from azureml.core import Workspace, Experiment, Environment, Dataset
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration


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
            "azureml-core",
            "azureml-dataset-runtime",
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
        compute_target = ws.compute_targets["cpu-cluster"]
        print("Using compute target:", compute_target.name)

        # Get registered dataset
        dataset = Dataset.get_by_name(ws, name="retail_dataset")
        print("Using dataset:", dataset.name)

        # Create environment
        env = create_environment(ws)
        print("Created environment:", env.name)

        # Create run configuration
        run_config = RunConfiguration()
        run_config.environment = env

        # Create output directories for models
        sqlgen_output = PipelineData(
            name="sqlgen_model", datastore=ws.get_default_datastore()
        )
        intent_output = PipelineData(
            name="intent_model", datastore=ws.get_default_datastore()
        )

        # Create training steps
        sqlgen_step = PythonScriptStep(
            name="train_sqlgen",
            script_name="train_sqlgen_t5_local.py",
            source_directory="code",
            compute_target=compute_target,
            inputs=[dataset.as_named_input("sqlgen_data")],
            outputs=[sqlgen_output],
            arguments=[
                "--input_data",
                "${{inputs.sqlgen_data}}",
                "--output_dir",
                "${{outputs.sqlgen_model}}",
            ],
            runconfig=run_config,
            allow_reuse=False,
        )

        intent_step = PythonScriptStep(
            name="train_intent",
            script_name="train_intent_classifier_local.py",
            source_directory="code",
            compute_target=compute_target,
            inputs=[dataset.as_named_input("intent_data")],
            outputs=[intent_output],
            arguments=[
                "--input_data",
                "${{inputs.intent_data}}",
                "--output_dir",
                "${{outputs.intent_model}}",
            ],
            runconfig=run_config,
            allow_reuse=False,
        )

        # Create pipeline
        pipeline = Pipeline(workspace=ws, steps=[sqlgen_step, intent_step])

        # Submit pipeline
        experiment = Experiment(ws, "retailgenie-training")
        pipeline_run = experiment.submit(pipeline)
        print("Pipeline submitted. Run ID:", pipeline_run.id)

        # Wait for completion
        pipeline_run.wait_for_completion()
        print("Pipeline completed!")

        # Publish pipeline
        published_pipeline = pipeline_run.publish_pipeline(
            name="retailgenie-training-pipeline",
            description="Training pipeline for RetailGenie models",
            version="1.0",
        )
        print("Pipeline published. Pipeline ID:", published_pipeline.id)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
