from azureml.core import Workspace
<<<<<<< HEAD
import os

# Get subscription ID from config
with open('config.json', 'r') as f:
    import json
    config = json.load(f)
    subscription_id = config['subscription_id']
    resource_group = config['resource_group']
    workspace_name = config['workspace_name']
    location = config['location']
=======

# Get subscription ID from config
with open("config.json", "r") as f:
    import json

    config = json.load(f)
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    workspace_name = config["workspace_name"]
    location = config["location"]
>>>>>>> adcb741 (ci/cd updated)

try:
    # Try to get existing workspace
    ws = Workspace.from_config()
<<<<<<< HEAD
    print(f'Found existing workspace: {ws.name}')
except:
    print('Creating new workspace...')
=======
    print(f"Found existing workspace: {ws.name}")
except:
    print("Creating new workspace...")
>>>>>>> adcb741 (ci/cd updated)
    # Create new workspace
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        create_resource_group=True,
<<<<<<< HEAD
        location=location
    )
    print(f'Created new workspace: {ws.name}')

# Save workspace configuration
ws.write_config()
print('Workspace configuration saved to .azureml/config.json')
=======
        location=location,
    )
    print(f"Created new workspace: {ws.name}")

# Save workspace configuration
ws.write_config()
print("Workspace configuration saved to .azureml/config.json")
>>>>>>> adcb741 (ci/cd updated)
