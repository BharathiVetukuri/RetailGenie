from azureml.core import Workspace
import os

# Get subscription ID from config
with open('config.json', 'r') as f:
    import json
    config = json.load(f)
    subscription_id = config['subscription_id']
    resource_group = config['resource_group']
    workspace_name = config['workspace_name']
    location = config['location']

try:
    # Try to get existing workspace
    ws = Workspace.from_config()
    print(f'Found existing workspace: {ws.name}')
except:
    print('Creating new workspace...')
    # Create new workspace
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        create_resource_group=True,
        location=location
    )
    print(f'Created new workspace: {ws.name}')

# Save workspace configuration
ws.write_config()
print('Workspace configuration saved to .azureml/config.json')
