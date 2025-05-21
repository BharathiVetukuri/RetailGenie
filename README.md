# 🛒 RetailGenie – An LLM-Powered BI Assistant

RetailGenie is an end-to-end deep learning + MLOps project that allows retail managers to ask questions in natural language and get:

- Generated SQL
- Predicted intent
- Tabular + chart-based insights

## 📂 Project Structure

- `data/` – Retail dataset (NL, intent, SQL) and schema
- `code/` – Training scripts for SQL generation (T5) and intent classification (BERT)
- `mlops/` – Metric logging, dashboards, and drift detection
- `pipelines/` – Vertex AI / Databricks pipeline definitions
- `ui/` – Gradio-based interface
- `report/` – Final paper, slides, and demo video
- `notebooks/` – Training notebooks (optional)

## 📈 Key Features

- LLM fine-tuning (`T5`, `DistilBERT`)
- Full CI/CD and auto-deploy with Vertex AI or Databricks
- Real-time inference + chart generation
- BLEU, accuracy, confusion matrix visualizations

## Project Artifacts

**Youtube Video Link**: https://youtu.be/YXPZawbxc3s 

**Colab Link**:

**Project Report Link**: https://docs.google.com/document/d/1DFvYuQQcZ2NcxfbIaWllHvJqroN-yr1INOVunanDdl8/edit?tab=t.0

**Slide Deck Link**:

## Screenshots

All our project artifacts and application related screenshots are present in this folder: 

## 📊 Dataset Sample

See `data/retail_dataset.csv` and `data/retail_schema.sql`

## 💻 Running the App Locally

To quickly run RetailGenie on your local machine, follow these steps:

### 1. Set Up Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the T5 SQL Generator Model

```bash
python code/train_sqlgen_t5_local.py
```

This will train the T5 model to generate SQL queries from natural language questions and save it to the `model_sqlgen_t5` directory.

### 4. Train the Intent Classifier Model

```bash
python code/train_intent_classifier_local.py
```

This will train the DistilBERT model to classify the intent of queries and save it to the `model_intent_classifier` directory.

### 5. Launch the Gradio App Interface

```bash
python ui/gradio_app.py
```

This will start the Gradio web interface where you can ask retail questions and get SQL queries, visualizations, and insights.

## Important Note

The model files are not included in this repository. You must train the models (steps 3-4) before running the application.

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- Python 3.9 or higher
- GCP project with Vertex AI + Artifact Registry enabled
- Google Cloud SDK installed + authenticated:

````bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

## Setup Environment

git clone https://github.com/YOUR_USERNAME/RetailGenie.git
cd RetailGenie

# Optional: set up virtual environment
python -m venv venv
source venv/bin/activate

# Install SDKs
pip install -r requirements.txt


#### Contents of requirements.txt:
google-cloud-aiplatform
google-cloud-storage
transformers
datasets
scikit-learn
pandas
torch

# 🏗️ Build & Push Containers
## 🧠 SQLGen

cd containers/train-sqlgen
docker build --no-cache -t us-central1-docker.pkg.dev/YOUR_PROJECT/retailgenie/retailgenie-train-sqlgen:latest .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/retailgenie/retailgenie-train-sqlgen:latest
## 🧠 Intent Classifier

cd containers/train-intent
docker build --no-cache -t us-central1-docker.pkg.dev/YOUR_PROJECT/retailgenie/retailgenie-train-intent:latest .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/retailgenie/retailgenie-train-intent:latest


# 📦 Submit the Vertex AI Pipeline
## 1. Compile the pipeline

cd pipelines
python compile_pipeline.py
## 2. Submit training-only pipeline

cd ..
python submit_pipeline.py

## 📁 Models will be saved to:

gs://your-bucket/model-outputs/sqlgen/

gs://your-bucket/model-outputs/intent/

## ☁️ Deploy Models from Local Machine
Once training is complete:

python deploy_from_local.py

## This script will:

Upload both models to Vertex AI

Create endpoints

Deploy with traffic split

## 🔗 Endpoints will be created in Vertex AI
You can find them under:
https://console.cloud.google.com/vertex-ai/endpoints

# RetailGenie Azure ML CI/CD Pipeline

This repository contains a CI/CD pipeline that automates the deployment of RetailGenie models to Azure Machine Learning.

## Pipeline Architecture

The CI/CD pipeline automates the following steps:

1. **Training Pipeline Submission**: Submits a pipeline to Azure ML that trains the RetailGenie models.
2. **Model Registration**: Registers the trained models in Azure ML Model Registry.
3. **Model Deployment**: Deploys the trained models as endpoints for inference.

## Components

- **run_pipeline.py**: Defines and submits the Azure ML pipeline.
- **register_model.py**: Registers trained models in Azure ML.
- **deploy_model.py**: Deploys registered models as endpoints.
- **score.py**: Scoring script used for model inference.
- **.github/workflows/azure-ml-pipeline.yml**: GitHub Actions workflow that orchestrates the CI/CD process.

## Requirements

To use this CI/CD pipeline, you need:

1. An Azure subscription
2. Azure ML workspace ("retailgenie-workspace")
3. Azure resource group ("retailgenie-rg")
4. Compute target set up in Azure ML ("retailgenie-cluster")
5. GitHub repository with the proper Azure credentials set up as secrets

## GitHub Secrets Required

The following secrets need to be set up in your GitHub repository:

- `AZURE_CREDENTIALS`: Service principal credentials for Azure authentication
- `AZURE_SUBSCRIPTION_ID`: Azure subscription ID
- `AZURE_CLIENT_ID`: Service principal client ID
- `AZURE_CLIENT_SECRET`: Service principal client secret
- `AZURE_TENANT_ID`: Azure tenant ID

## How to Set Up Azure Service Principal

1. Create a service principal with contributor access to your Azure ML workspace:

```bash
az ad sp create-for-rbac --name "retailgenie-sp" --role contributor \
                         --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} \
                         --sdk-auth
````

2. Use the output JSON as the value for the `AZURE_CREDENTIALS` secret in GitHub.

## Viewing Results in Azure ML

After the pipeline runs successfully:

1. **Pipelines**: View your pipeline runs in Azure ML Studio under "Pipelines".
2. **Models**: Find your registered models under "Models".
3. **Endpoints**: Access your deployed models under "Endpoints".

## Manual Execution

You can manually trigger the CI/CD pipeline from the GitHub Actions tab using the "workflow_dispatch" event.

## Customization

To customize the pipeline for your specific needs:

1. Modify `run_pipeline.py` to adjust training parameters or steps.
2. Update `score.py` to match your model's input/output requirements.
3. Edit `.github/workflows/azure-ml-pipeline.yml` to change the CI/CD workflow.

```

```

## Team Members and Contributions

**Soumya Bharathi Vetukuri**

Key Contributions:

* Designed the overall Vertex AI pipeline architecture for training and deployment

* Wrote pipeline.py and integrated all pipeline steps (T5 training, Intent Classifier training, and deployment)

* Implemented and tested submit_pipeline.py and compile_pipeline.py

* Troubleshot MLOps errors including auth scopes, GCS pathing, and container failures

* Validated success criteria for pipeline completion and GCS model outputs

**Shubham Jaysukhbhai Kothiya**

Key Contributions:

* Preprocessed and curated the retail Q&A dataset

* Implemented train_sqlgen_t5.py using Hugging Face T5 for natural language to SQL generation

* Wrote train_intent_classifier.py using DistilBERT for intent classification

* Tuned model hyperparameters, added tokenization logic, and structured Hugging Face Trainer configs

* Managed local training experiments and transitioned models to cloud containers

**Rutuja Patil**

Key Contributions:

* Designed Dockerfiles for all three containers: train-sqlgen, train-intent, and deploy

* Managed Artifact Registry and pushed all container images using docker build and docker push

* Debugged container runtime failures (e.g., missing packages like gcsfs, sentencepiece)

* Integrated GCS upload logic into the training scripts using google-cloud-storage

* Ensured containers authenticated properly inside Vertex AI jobs

**Yugm Patel**

Key Contributions:

* Developed the local deploy_from_local.py script for deploying trained models to Vertex AI

* Handled aiplatform.Model.upload() and Endpoint.create() logic

* Set up environment-level authentication using ADC and verified GCP permissions

* Verified endpoint creation and tested response handling


