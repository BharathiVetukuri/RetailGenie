import argparse
import os
from google.cloud import aiplatform

# Ensure correct auth scope inside Vertex AI container
os.environ["GOOGLE_AUTH_SCOPES"] = "https://www.googleapis.com/auth/cloud-platform"

def upload_and_deploy_model(model_path, display_name, project, region, endpoint_name):
    aiplatform.init(project=project, location=region)

    # Upload model
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
    )
    model.wait()

    # Create or get endpoint
    endpoint = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    if endpoint:
        endpoint = endpoint[0]
        print(f"Using existing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        endpoint.wait()
        print(f"Created new endpoint: {endpoint.resource_name}")

    # Deploy model
    deployed_model = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{display_name}-deployed",
        traffic_split={"0": 100},
        machine_type="n1-standard-2"
    )

    print(f"✅ Model '{display_name}' deployed to endpoint '{endpoint.display_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql_model_path", required=True)
    parser.add_argument("--intent_model_path", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    args = parser.parse_args()

    print("🚀 Deploying SQL Generator...")
    upload_and_deploy_model(
        args.sql_model_path,
        display_name="RetailGenie-SQLGen",
        project=args.project,
        region=args.region,
        endpoint_name="retailgenie-sqlgen-endpoint"
    )

    print("🚀 Deploying Intent Classifier...")
    upload_and_deploy_model(
        args.intent_model_path,
        display_name="RetailGenie-IntentClassifier",
        project=args.project,
        region=args.region,
        endpoint_name="retailgenie-intent-endpoint"
    )
