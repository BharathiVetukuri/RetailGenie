from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
import yaml
import os
import logging
import mlflow
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("azure_config.yaml", "r") as file:
        return yaml.safe_load(file)["azure"]


class ModelDeployer:
    def __init__(self):
        self.config = load_config()
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=self.config["resource_group"],
            workspace_name=self.config["workspace"]["name"],
        )

        # Initialize MLflow
        mlflow.set_tracking_uri(
            f"azureml://{self.config['workspace']['name']}.workspace.azureml.net"
        )

    def register_model(self, model_path, model_name):
        """Register model in Azure ML"""
        try:
            model = Model(
                path=model_path,
                name=model_name,
                description=f"RetailGenie model registered on {datetime.now()}",
            )

            registered_model = self.ml_client.models.create_or_update(model)
            logger.info(f"Model registered: {registered_model.name}")
            return registered_model
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise

    def create_endpoint(self, endpoint_name):
        """Create online endpoint"""
        try:
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description="RetailGenie model endpoint",
                auth_mode="key",
            )

            self.ml_client.online_endpoints.begin_create_or_update(endpoint)
            logger.info(f"Endpoint created: {endpoint_name}")
        except Exception as e:
            logger.error(f"Failed to create endpoint: {str(e)}")
            raise

    def deploy_model(self, model_name, endpoint_name, deployment_name):
        """Deploy model to endpoint"""
        try:
            # Get the registered model
            model = self.ml_client.models.get(name=model_name, version="latest")

            # Create deployment
            deployment = ManagedOnlineDeployment(
                name=deployment_name,
                endpoint_name=endpoint_name,
                model=model,
                environment="retailgenie-inference-env",
                code_configuration=CodeConfiguration(
                    code="ui/", scoring_script="gradio_app.py"
                ),
                instance_type="Standard_DS1_v2",  # Free tier eligible
                instance_count=1,
                app_insights_enabled=True,
                request_settings={
                    "request_timeout_ms": 90000,
                    "max_concurrent_requests_per_instance": 1,
                },
            )

            self.ml_client.online_deployments.begin_create_or_update(deployment)
            logger.info(f"Model deployed: {deployment_name}")
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            raise

    def update_traffic(self, endpoint_name, deployment_name, traffic_percentage=100):
        """Update traffic distribution"""
        try:
            self.ml_client.online_endpoints.update_traffic(
                name=endpoint_name, traffic={deployment_name: traffic_percentage}
            )
            logger.info(f"Traffic updated for {endpoint_name}")
        except Exception as e:
            logger.error(f"Failed to update traffic: {str(e)}")
            raise

    def deploy_new_version(self, model_path, model_name, endpoint_name):
        """Deploy new version of the model"""
        try:
            # Register new model version
            model = self.register_model(model_path, model_name)

            # Create new deployment
            deployment_name = f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.deploy_model(model_name, endpoint_name, deployment_name)

            # Gradually shift traffic to new deployment
            self.update_traffic(endpoint_name, deployment_name, 50)

            logger.info(f"New version deployed: {deployment_name}")
            return deployment_name
        except Exception as e:
            logger.error(f"Failed to deploy new version: {str(e)}")
            raise


def main():
    deployer = ModelDeployer()

    # Deploy initial version
    model_path = "models/"
    model_name = "retailgenie-model"
    endpoint_name = "retailgenie-endpoint"

    try:
        # Create endpoint if it doesn't exist
        deployer.create_endpoint(endpoint_name)

        # Deploy model
        deployment_name = deployer.deploy_new_version(
            model_path=model_path, model_name=model_name, endpoint_name=endpoint_name
        )

        print(f"Deployment completed successfully: {deployment_name}")
    except Exception as e:
        print(f"Deployment failed: {str(e)}")


if __name__ == "__main__":
    main()
