from azureml.core import Workspace, Dataset
import os


def register_dataset():
    # Get workspace
    ws = Workspace.from_config()
    print("Connected to workspace:", ws.name)

    try:
        # Get or create default datastore
        default_datastore = ws.get_default_datastore()
        print(f"Using datastore: {default_datastore.name}")

        # Upload the dataset to the datastore
        data_path = os.path.join("data", "testing_sql_data.csv")
        target_path = "retail_data"

        # Upload the file to the datastore
        default_datastore.upload_files(
            files=[data_path], target_path=target_path, overwrite=True
        )
        print(f"Uploaded dataset to {target_path}")

        # Create a FileDataset from the datastore
        dataset = Dataset.File.from_files(
            path=(default_datastore, os.path.join(target_path, "testing_sql_data.csv"))
        )

        # Register the dataset
        dataset = dataset.register(
            workspace=ws,
            name="retail_dataset",
            description="Retail dataset for SQL generation and intent classification",
            create_new_version=True,
        )

        print(f"Dataset registered successfully: {dataset.name}")
        print(f"Dataset version: {dataset.version}")

    except Exception as e:
        print(f"Error registering dataset: {str(e)}")
        raise


if __name__ == "__main__":
    register_dataset()
