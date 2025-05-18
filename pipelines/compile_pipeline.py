import os
from kfp.v2 import compiler
from pipeline import retailgenie_pipeline

# Get project root directory (one level up from current directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Output file path for the compiled pipeline
output_path = os.path.join(os.path.dirname(__file__), "retailgenie_pipeline.json")

# Compile the KFP pipeline to a JSON file
compiler.Compiler().compile(
    pipeline_func=retailgenie_pipeline, package_path=output_path
)
