from azure.ai.ml import MLClient
from azure.ai.ml.entities import ComputeInstance, AmlCompute, Environment
from azure.identity import DefaultAzureCredential
import yaml
import os


def load_config():
    with open("azure_config.yaml", "r") as file:
        return yaml.safe_load(file)["azure"]


def setup_azure_ml():
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

    # Create compute instance for training
    training_compute = ComputeInstance(
        name=config["compute"]["training"]["name"],
        type="computeinstance",
        size=config["compute"]["training"]["size"],
        idle_time_before_shutdown=3600,  # Shutdown after 1 hour of inactivity
    )
    ml_client.compute.begin_create_or_update(training_compute)

    # Create compute cluster for inference
    inference_compute = AmlCompute(
        name=config["compute"]["inference"]["name"],
        type="amlcompute",
        size=config["compute"]["inference"]["size"],
        min_instances=0,  # Scale to zero when not in use
        max_instances=1,  # Free tier limit
        idle_time_before_scale_down=300,  # Scale down after 5 minutes
    )
    ml_client.compute.begin_create_or_update(inference_compute)

    # Create environment for training
    training_env = Environment(
        name="retailgenie-training-env",
        description="Environment for training RetailGenie models",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
    ml_client.environments.create_or_update(training_env)

    # Create environment for inference
    inference_env = Environment(
        name="retailgenie-inference-env",
        description="Environment for serving RetailGenie models",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
    ml_client.environments.create_or_update(inference_env)

    print("Azure ML setup completed successfully!")


if __name__ == "__main__":
    setup_azure_ml()
