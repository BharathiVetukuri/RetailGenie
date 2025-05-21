from azure.monitor import MonitorClient
from azure.identity import DefaultAzureCredential
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("azure_config.yaml", "r") as file:
        return yaml.safe_load(file)["azure"]


class AlertSystem:
    def __init__(self):
        self.config = load_config()
        self.credential = DefaultAzureCredential()
        self.monitor_client = MonitorClient(self.credential)

        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.alert_recipients = os.getenv("ALERT_RECIPIENTS", "").split(",")

    def send_email_alert(self, subject, message, priority="normal"):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_username
            msg["To"] = ", ".join(self.alert_recipients)
            msg["Subject"] = f"[{priority.upper()}] {subject}"

            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Alert email sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send alert email: {str(e)}")

    def create_metric_alert(self, metric_name, threshold, operator, severity=2):
        """Create Azure Monitor metric alert"""
        try:
            alert_rule = {
                "location": self.config["location"],
                "properties": {
                    "description": f"Alert when {metric_name} {operator} {threshold}",
                    "severity": severity,
                    "enabled": True,
                    "scopes": [
                        f"/subscriptions/{os.getenv('AZURE_SUBSCRIPTION_ID')}/resourceGroups/{self.config['resource_group']}/providers/Microsoft.MachineLearningServices/workspaces/{self.config['workspace']['name']}"
                    ],
                    "evaluationFrequency": "PT1H",
                    "windowSize": "PT5M",
                    "criteria": {
                        "odata.type": "Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria",
                        "allOf": [
                            {
                                "name": "Metric1",
                                "metricName": metric_name,
                                "operator": operator,
                                "threshold": threshold,
                                "timeAggregation": "Average",
                            }
                        ],
                    },
                    "actions": [],
                },
            }

            self.monitor_client.metric_alerts.create_or_update(
                resource_group_name=self.config["resource_group"],
                rule_name=f"{metric_name}-alert",
                parameters=alert_rule,
            )

            logger.info(f"Created metric alert for {metric_name}")
        except Exception as e:
            logger.error(f"Failed to create metric alert: {str(e)}")

    def alert_on_drift(self, drift_report):
        """Alert on data drift detection"""
        if drift_report["drift_detected"]:
            subject = "Data Drift Detected"
            message = f"""
            Data drift has been detected in the model.
            
            Model: {drift_report['model_name']}
            Version: {drift_report['version']}
            Timestamp: {drift_report['timestamp']}
            
            Drift Details:
            {drift_report['drift_details']}
            """
            self.send_email_alert(subject, message, priority="high")

    def alert_on_performance(self, performance_report):
        """Alert on performance degradation"""
        if (
            performance_report["metrics"].get("accuracy", 1.0) < 0.8
        ):  # Example threshold
            subject = "Model Performance Degradation"
            message = f"""
            Model performance has degraded below threshold.
            
            Model: {performance_report['model_name']}
            Version: {performance_report['version']}
            Current Accuracy: {performance_report['metrics'].get('accuracy', 'N/A')}
            """
            self.send_email_alert(subject, message, priority="high")

    def setup_alerts(self):
        """Set up all monitoring alerts"""
        # Set up metric alerts
        self.create_metric_alert(
            metric_name="ModelLatency",
            threshold=1000,  # 1 second
            operator="GreaterThan",
        )

        self.create_metric_alert(
            metric_name="ErrorRate", threshold=0.05, operator="GreaterThan"  # 5%
        )

        logger.info("Alert system setup completed")


def main():
    alert_system = AlertSystem()
    alert_system.setup_alerts()

    # Example usage
    drift_report = {
        "model_name": "retailgenie-model",
        "version": "1.0.0",
        "drift_detected": True,
        "drift_details": {"feature1": {"drift_detected": True}},
        "timestamp": "2024-03-20T10:00:00",
    }

    alert_system.alert_on_drift(drift_report)


if __name__ == "__main__":
    main()
