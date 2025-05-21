from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Azure subscription details
subscription_id = "bd063a5e-8319-4484-8621-5efd18d9f79d"
resource_group = "retailgenie-rg"
workspace_name = "retailgenie-workspace"
location = "eastus"

try:
    # Try to get existing workspace
    ws = Workspace.from_config()
    print(f"Found existing workspace: {ws.name}")
except:
    print("Creating new workspace...")
    # Create new workspace
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        create_resource_group=True,
        location=location,
    )
    print(f"Created new workspace: {ws.name}")

# Save workspace configuration
ws.write_config()
print("Workspace configuration saved to .azureml/config.json")

# Set up compute target
compute_name = "retailgenie-cluster"

try:
    # Check if compute target exists
    compute_target = AmlCompute(ws, compute_name)
    print(f"Found existing compute target: {compute_name}")
except ComputeTargetException:
    print(f"Creating new compute target: {compute_name}")

    # Create compute target configuration
    # Using Standard_NV6 which is supported in eastus region
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_NV6",  # GPU-enabled VM supported in eastus
        min_nodes=1,
        max_nodes=2,
        idle_seconds_before_scaledown=300,
    )

    # Create compute target
    compute_target = AmlCompute.create(ws, compute_name, provisioning_config)

    # Wait for completion
    compute_target.wait_for_completion(show_output=True)

print(f"Compute target {compute_name} is ready")
