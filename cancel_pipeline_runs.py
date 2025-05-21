from azureml.core import Workspace, Experiment
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cancel_runs(experiment_name=None):
    """Cancel all running pipeline runs in the specified experiment."""
    try:
        # Connect to workspace
        logger.info("Connecting to workspace...")
        ws = Workspace.from_config()
        logger.info(f"Connected to workspace: {ws.name}")

        # Get all experiments or a specific one
        if experiment_name:
            experiments = [Experiment(ws, experiment_name)]
            logger.info(f"Looking for runs in experiment: {experiment_name}")
        else:
            experiments = ws.experiments.values()
            logger.info(
                f"Looking for runs in all experiments ({len(experiments)} total)"
            )

        # Track the number of runs canceled
        canceled_count = 0

        # Process each experiment
        for experiment in experiments:
            logger.info(f"Checking experiment: {experiment.name}")

            # Get all running runs
            runs = list(experiment.get_runs())
            running_runs = [run for run in runs if run.status == "Running"]

            if not running_runs:
                logger.info(f"No running runs found in {experiment.name}")
                continue

            logger.info(f"Found {len(running_runs)} running runs in {experiment.name}")

            # Cancel each run
            for run in running_runs:
                try:
                    submit_time = (
                        run.submit_time if hasattr(run, "submit_time") else "unknown"
                    )
                    logger.info(f"Canceling run {run.id} (submitted: {submit_time})")
                    run.cancel()
                    canceled_count += 1
                except Exception as e:
                    logger.error(f"Error canceling run {run.id}: {str(e)}")

        if canceled_count > 0:
            logger.info(f"Successfully canceled {canceled_count} runs")
        else:
            logger.info("No running runs found to cancel")

    except Exception as e:
        logger.error(f"Error canceling runs: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Cancel running Azure ML pipeline runs"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Name of the experiment to cancel runs for (default: all experiments)",
    )
    args = parser.parse_args()

    cancel_runs(args.experiment)


if __name__ == "__main__":
    main()
