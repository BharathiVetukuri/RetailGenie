from azureml.core import Workspace
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="List all datasets in an Azure ML workspace"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed information about each dataset",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (text or json)",
    )
    args = parser.parse_args()

    try:
        # Connect to workspace using config file
        print("Loading Azure ML workspace...", file=sys.stderr)
        ws = Workspace.from_config()
        print(f"Connected to workspace: {ws.name}", file=sys.stderr)

        # Get all datasets in the workspace
        datasets = ws.datasets

        if args.output == "json":
            # For JSON output
            result = []
            for name, dataset in datasets.items():
                dataset_info = {
                    "name": name,
                    "version": dataset.version,
                    "description": dataset.description,
                    "created_date": str(dataset.created_date),
                    "type": type(dataset).__name__,
                }
                if args.details:
                    if hasattr(dataset, "tags"):
                        dataset_info["tags"] = dataset.tags
                    if hasattr(dataset, "datastore"):
                        dataset_info["datastore"] = dataset.datastore.name
                result.append(dataset_info)
            print(json.dumps(result, indent=2))
        else:
            # For text output
            if len(datasets) == 0:
                print("No datasets found in this workspace")
            else:
                if args.details:
                    for name, dataset in datasets.items():
                        print(f"Name: {name}")
                        print(f"  - Version: {dataset.version}")
                        print(f"  - Description: {dataset.description}")
                        print(f"  - Created: {dataset.created_date}")
                        print(f"  - Type: {type(dataset).__name__}")
                        if hasattr(dataset, "tags") and dataset.tags:
                            print(f"  - Tags: {dataset.tags}")
                        print("--------------------------------------")
                else:
                    # Simple list of names
                    for name in datasets.keys():
                        print(name)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
