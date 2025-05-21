from azureml.core import Workspace, Experiment
from azureml.core.environment import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
import os

# Check if training script exists, create if not
script_path = "simple_train.py"
if not os.path.exists(script_path):
    print(f"Creating {script_path}...")
    with open(script_path, "w") as f:
        f.write(
            """
from azureml.core import Run
import argparse
import os

# Get context and parse arguments
run = Run.get_context()
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="World")
args = parser.parse_args()

# Print a greeting
print(f"Hello, {args.name}!")

# Log a metric
run.log("greeting_sent", True)

# Write a dummy output file
os.makedirs("outputs", exist_ok=True)
with open(os.path.join("outputs", "output.txt"), "w") as f:
    f.write(f"Hello, {args.name}!")

print("Done!")
"""
        )

# Connect to workspace
print("Connecting to workspace...")
ws = Workspace.from_config()
print(f"Connected to workspace: {ws.name}")

# Get compute target
compute_targets = list(ws.compute_targets)
print("Available compute targets:")
for ct in compute_targets:
    print(f"  - {ct}")

compute_name = input(
    f"Enter compute target name [{compute_targets[0] if compute_targets else 'cpu-cluster'}]: "
)
if not compute_name and compute_targets:
    compute_name = compute_targets[0]
elif not compute_name:
    compute_name = "cpu-cluster"

# Get compute target
compute_target = ws.compute_targets[compute_name]
print(f"Using compute target: {compute_name}")

# Create a basic environment
env = Environment(name="basic-env")
env.python.user_managed_dependencies = True
env.docker.enabled = True

# Create run config
run_config = RunConfiguration()
run_config.environment = env

# Define a simple step with no input or output bindings
step = PythonScriptStep(
    name="hello_step",
    script_name=script_path,
    compute_target=compute_target,
    source_directory=".",
    runconfig=run_config,
    arguments=["--name", "Azure"],
    allow_reuse=False,
)

# Build the pipeline
pipeline = Pipeline(workspace=ws, steps=[step])

# Create experiment and submit
experiment = Experiment(ws, "simple-hello-experiment")
run = experiment.submit(pipeline)

print(f"Pipeline is submitted with run ID: {run.id}")
print(f"View run details at: {run.get_portal_url()}")

# Wait for the pipeline to complete
print("Waiting for pipeline to complete...")
run.wait_for_completion(show_output=True)

print("Pipeline run completed!")
