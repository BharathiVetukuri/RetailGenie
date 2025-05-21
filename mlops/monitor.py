from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os


def load_config():
    with open("azure_config.yaml", "r") as file:
        return yaml.safe_load(file)["azure"]


class ModelMonitor:
    def __init__(self):
        self.config = load_config()
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=self.config["resource_group"],
            workspace_name=self.config["workspace"]["name"],
        )

        # Initialize MLflow
        mlflow.set_tracking_uri(
            f"azureml://{self.config['workspace']['name']}.workspace.azureml.net"
        )

    def check_model_performance(self, model_name, start_date=None, end_date=None):
        """Check model performance metrics"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()

        # Get model metrics from MLflow
        metrics = mlflow.search_runs(
            experiment_names=[model_name],
            filter_string=f"start_time >= '{start_date}' AND end_time <= '{end_date}'",
        )

        return metrics

    def detect_data_drift(self, reference_data, current_data, feature_columns):
        """Detect data drift using statistical tests"""
        drift_results = {}

        for column in feature_columns:
            # Perform Kolmogorov-Smirnov test
            from scipy import stats

            ks_stat, p_value = stats.ks_2samp(
                reference_data[column], current_data[column]
            )

            drift_results[column] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "drift_detected": p_value < 0.05,
            }

        return drift_results

    def monitor_model_health(self, model_name):
        """Monitor overall model health"""
        # Get latest model version
        model = self.ml_client.models.get(name=model_name, version="latest")

        # Get model metrics
        metrics = self.check_model_performance(model_name)

        # Check for data drift
        # This is a placeholder - you'll need to implement actual data collection
        reference_data = pd.DataFrame()  # Your reference data
        current_data = pd.DataFrame()  # Your current data
        drift_results = self.detect_data_drift(
            reference_data,
            current_data,
            feature_columns=["feature1", "feature2"],  # Your feature columns
        )

        # Compile health report
        health_report = {
            "model_name": model_name,
            "version": model.version,
            "metrics": metrics.to_dict() if not metrics.empty else {},
            "drift_detected": any(
                result["drift_detected"] for result in drift_results.values()
            ),
            "drift_details": drift_results,
            "timestamp": datetime.now().isoformat(),
        }

        return health_report

    def trigger_retraining(self, model_name, reason):
        """Trigger model retraining if needed"""
        # Log retraining trigger
        mlflow.log_param("retraining_triggered", True)
        mlflow.log_param("retraining_reason", reason)

        # You can add your retraining logic here
        # For example, calling your training pipeline
        print(f"Retraining triggered for {model_name}. Reason: {reason}")


def main():
    monitor = ModelMonitor()

    # Monitor model health
    health_report = monitor.monitor_model_health("retailgenie-model")

    # Check if retraining is needed
    if health_report["drift_detected"]:
        monitor.trigger_retraining("retailgenie-model", "Data drift detected")

    print("Model monitoring completed!")


if __name__ == "__main__":
    main()
