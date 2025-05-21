from azureml.core import Workspace, Experiment, Environment
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Connect to workspace
        logger.info("Connecting to workspace...")
        ws = Workspace.from_config()
        logger.info(f"Connected to workspace: {ws.name}")

        # Get compute target - list available options
        logger.info("Available compute targets:")
        for name in ws.compute_targets:
            logger.info(f"- {name}")

        compute_name = (
            input("Enter compute target name (default: cpu-cluster): ") or "cpu-cluster"
        )
        compute_target = ws.compute_targets[compute_name]
        logger.info(f"Using compute target: {compute_target.name}")

        # Create a simple environment
        logger.info("Creating a minimal environment...")
        env = Environment(name="minimal-env")
        conda_deps = CondaDependencies.create(
            python_version="3.7", pip_packages=["azureml-defaults"]
        )
        env.python.conda_dependencies = conda_deps

        # Create run config
        run_config = RunConfiguration()
        run_config.environment = env

        # Create output data
        model_output = PipelineData(
            name="model_output", datastore=ws.get_default_datastore()
        )

        # Create a simple step with no input dataset
        train_step = PythonScriptStep(
            name="quick_train",
            script_name="quick_dummy_train.py",
            source_directory=".",
            compute_target=compute_target,
            outputs=[model_output],
            arguments=["--output_dir", model_output],
            runconfig=run_config,
        )

        # Create pipeline
        pipeline = Pipeline(workspace=ws, steps=[train_step])

        # Pipeline name
        pipeline_name = "Quick-Training-Pipeline"

        # Publish pipeline
        published_pipeline = pipeline.publish(
            name=pipeline_name, description="Quick training pipeline", version="1.0"
        )
        logger.info(f"Pipeline published. ID: {published_pipeline.id}")

        # Submit for execution
        logger.info("Submitting pipeline for execution...")
        experiment = Experiment(ws, "quick-training")
        pipeline_run = experiment.submit(pipeline)
        run_id = pipeline_run.id
        logger.info(f"Pipeline submitted. Run ID: {run_id}")
        logger.info(f"View in Azure ML Studio: {pipeline_run.get_portal_url()}")

        return run_id

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    run_id = main()
    print(f"Pipeline run ID: {run_id}")
