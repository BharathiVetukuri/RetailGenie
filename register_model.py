from azureml.core import Workspace, Run
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def register_model_from_output(run_id, model_path, model_name):
    """Register the model from a completed run."""
    try:
        # Connect to workspace
        ws = Workspace.from_config()
        logger.info(f"Connected to workspace: {ws.name}")

        # Get the run using run ID
        run = Run.get(ws, run_id)
        logger.info(f"Run found: {run.id}, status: {run.status}")

        # Register the model
        model = run.register_model(
            model_name=model_name,
            model_path=model_path,
            description=f"RetailGenie {model_name} model",
            tags={"source_dir": ".", "run_id": run_id, "model_type": model_name},
        )

        logger.info(f"Model registered: {model.name}, version: {model.version}")
        return model

    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID from pipeline run"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model in the run outputs",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name to register the model under"
    )

    args = parser.parse_args()

    register_model_from_output(args.run_id, args.model_path, args.model_name)


if __name__ == "__main__":
    main()
