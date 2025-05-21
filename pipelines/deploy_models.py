from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment


def get_workspace():
    """Get Azure ML workspace."""
    return Workspace.from_config()


def deploy_models(ws):
    """Deploy models to Azure ML endpoints."""
    # Get the latest model versions
    sql_model = Model(ws, name="retailgenie-sqlgen")
    intent_model = Model(ws, name="retailgenie-intent")

    # Create environment
    env = Environment("retailgenie-env")
    env.python.conda_dependencies = Environment.from_conda_specification(
        name="retailgenie-env", file_path="environment.yml"
    )

    # Create inference configuration
    inference_config = InferenceConfig(
        entry_script="code/inference.py", environment=env
    )

    # Create deployment configuration
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1, memory_gb=4, auth_enabled=True, enable_app_insights=True
    )

    # Deploy service
    service = Model.deploy(
        workspace=ws,
        name="retailgenie-service",
        models=[sql_model, intent_model],
        inference_config=inference_config,
        deployment_config=deployment_config,
    )

    # Wait for deployment
    service.wait_for_deployment(show_output=True)

    print(f"Service deployed to: {service.scoring_uri}")
    print(f"Service state: {service.state}")

    return service


def main():
    """Main function to deploy models."""
    # Get workspace
    ws = get_workspace()

    # Deploy models
    service = deploy_models(ws)

    # Save service details
    with open("service_details.txt", "w") as f:
        f.write(f"Service URI: {service.scoring_uri}\n")
        f.write(f"Service State: {service.state}\n")
        f.write(f"Service ID: {service.id}\n")


if __name__ == "__main__":
    main()
