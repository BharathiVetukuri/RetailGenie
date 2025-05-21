from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException


def main():
    ws = Workspace.from_config()
    compute_name = "cpu-cluster"
    try:
        compute_target = AmlCompute(ws, compute_name)
        print(f"Found existing compute target: {compute_name}")
    except ComputeTargetException:
        print(f"Creating new compute target: {compute_name}")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="Standard_DS3_v2", min_nodes=0, max_nodes=2  # CPU VM
        )
        compute_target = AmlCompute.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)
    print(f"Compute target {compute_name} is ready")


if __name__ == "__main__":
    main()
