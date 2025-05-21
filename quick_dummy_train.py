import argparse
import os
import time
from azureml.core import Run


def main():
    # Get the run context
    run = Run.get_context()

    print("Starting super-quick training...")

    # Create the output directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Log a metric
    run.log("accuracy", 0.95)

    # Brief pause
    time.sleep(5)

    # Create a dummy model file
    with open(os.path.join(args.output_dir, "model.txt"), "w") as f:
        f.write("Dummy model file")

    print("Training completed quickly!")


if __name__ == "__main__":
    main()
