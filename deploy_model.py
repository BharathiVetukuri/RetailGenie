from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
import argparse
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def deploy_model(model_name, model_version, entry_script):
    """Deploy a registered model as a web service."""
    try:
        # Connect to workspace
        ws = Workspace.from_config()
        logger.info(f"Connected to workspace: {ws.name}")

        # Get the registered model
        model = Model(ws, name=model_name, version=model_version)
        logger.info(f"Model loaded: {model.name}, version: {model.version}")

        # Create inference config
        inference_config = InferenceConfig(
            entry_script=entry_script,
            source_directory=".",
            environment=get_deployment_environment(ws),
        )

        # Configure the deployment
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=1,
            auth_enabled=True,
            enable_app_insights=True,
            description=f"RetailGenie {model_name} API",
        )

        # Deploy the web service
        service_name = f"{model_name}-service"
        service = Model.deploy(
            workspace=ws,
            name=service_name,
            models=[model],
            inference_config=inference_config,
            deployment_config=deployment_config,
            overwrite=True,
        )

        # Wait for deployment to complete
        service.wait_for_deployment(show_output=True)

        # Get the scoring URI
        scoring_uri = service.scoring_uri
        logger.info("Model deployed successfully!")
        logger.info(f"Service name: {service.name}")
        logger.info(f"Scoring URI: {scoring_uri}")

        # Save endpoint info to file
        endpoint_info = {
            "service_name": service.name,
            "scoring_uri": scoring_uri,
            "model_name": model_name,
            "model_version": model_version,
        }

        with open(f"{model_name}_endpoint.json", "w") as f:
            json.dump(endpoint_info, f, indent=2)

        return service

    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise


def get_deployment_environment(ws):
    """Get or create the deployment environment."""
    from azureml.core import Environment
    from azureml.core.conda_dependencies import CondaDependencies

    # Create deployment environment
    env = Environment(name="retailgenie-deployment-env")

    # Create conda dependencies
    conda_deps = CondaDependencies.create(
        python_version="3.8",
        pip_packages=[
            "torch",
            "transformers",
            "pandas",
            "scikit-learn",
            "azureml-defaults",
            "flask",
            "gunicorn",
        ],
    )

    # Set conda dependencies
    env.python.conda_dependencies = conda_deps

    # Register environment
    env.register(workspace=ws)

    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the registered model"
    )
    parser.add_argument(
        "--model_version",
        type=int,
        required=True,
        help="Version of the registered model",
    )
    parser.add_argument(
        "--entry_script", type=str, required=True, help="Path to scoring script"
    )

    args = parser.parse_args()

    deploy_model(args.model_name, args.model_version, args.entry_script)


if __name__ == "__main__":
    main()
