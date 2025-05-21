from azureml.core import Workspace
from azureml.data.dataset_factory import TabularDatasetFactory
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def register_dataset():
    """Register a dataset in Azure ML workspace."""
    try:
        # Connect to workspace
        logger.info("Connecting to workspace...")
        ws = Workspace.from_config()
        logger.info(f"Connected to workspace: {ws.name}")

        # Ask for dataset details
        csv_path = (
            input("Enter local path to CSV file (default: data/retail_data.csv): ")
            or "data/retail_data.csv"
        )
        dataset_name = (
            input("Enter dataset name (default: retail_dataset): ") or "retail_dataset"
        )
        dataset_desc = (
            input("Enter dataset description (optional): ")
            or "Retail data for RetailGenie"
        )

        # Check if file exists
        if not os.path.exists(csv_path):
            logger.error(f"File not found: {csv_path}")
            return

        # Get default datastore
        datastore = ws.get_default_datastore()
        logger.info(f"Using default datastore: {datastore.name}")

        # Upload file to datastore
        logger.info(f"Uploading {csv_path} to datastore...")
        datastore.upload_files(
            files=[csv_path],
            target_path="datasets/retail_data",
            overwrite=True,
            show_progress=True,
        )

        # Create dataset from datastore path
        datastore_path = datastore.path(
            f"datasets/retail_data/{os.path.basename(csv_path)}"
        )
        logger.info(f"Creating dataset from {datastore_path}...")

        # Create tabular dataset
        dataset = TabularDatasetFactory.from_delimited_files(
            path=datastore_path,
            validate=True,
            include_path=False,
            infer_column_types=True,
            set_column_types=None,
        )

        # Register dataset
        dataset = dataset.register(
            workspace=ws,
            name=dataset_name,
            description=dataset_desc,
            create_new_version=True,
        )

        logger.info(f"Dataset registered: {dataset.name}, version: {dataset.version}")
        logger.info(
            "You can view this dataset in Azure ML Studio under Data → Datasets"
        )

        return dataset

    except Exception as e:
        logger.error(f"Error registering dataset: {str(e)}")
        raise


if __name__ == "__main__":
    register_dataset()
