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

**Slide Deck Link**: https://prezi.com/view/IsYkiUo37RbUuPnNzC5f/

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

# RetailGenie CI/CD to Azure ML

This repository contains code that trains and deploys a model to Azure Machine Learning as part of a CI/CD pipeline using GitHub Actions.

## Architecture

This project implements a modern MLOps approach:

1. **CI/CD with GitHub Actions**: The model is trained, tested, and deployed automatically when code is pushed.
2. **Model Training**: A simple model is trained directly in the GitHub Actions workflow.
3. **Model Registration**: The trained model is registered in the Azure ML Model Registry.
4. **Model Deployment**: The model is deployed as an Azure Container Instance (ACI) endpoint.
5. **Endpoint Testing**: The deployed endpoint is tested to ensure it's working correctly.

## Setup

### Prerequisites

To use this pipeline, you need:

1. An Azure subscription
2. An Azure Machine Learning workspace
3. GitHub repository with GitHub Actions enabled

### GitHub Secrets

You need to set up the following secrets in your GitHub repository:

- `AZURE_CREDENTIALS`: Service principal credentials for Azure authentication
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID

### Creating Azure Service Principal

1. Create a service principal with contributor access to your Azure ML workspace:

```bash
az ad sp create-for-rbac --name "retailgenie-sp" --role contributor \
                         --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} \
                         --sdk-auth
````

2. Use the output JSON as the value for the `AZURE_CREDENTIALS` secret in GitHub.

## Workflow

The GitHub Actions workflow performs the following steps:

1. **Environment Setup**: Sets up Python and installs necessary packages
2. **Model Training**: Trains a RandomForest classifier on sample data
3. **Model Registration**: Registers the trained model in Azure ML
4. **Scoring Script Creation**: Creates a scoring script for model deployment
5. **Model Deployment**: Deploys the model as an ACI endpoint
6. **Endpoint Testing**: Tests the deployed endpoint with sample data

## Viewing Results

After the workflow runs successfully:

1. **Models**: View your registered models in Azure ML Studio under "Models"
2. **Endpoints**: Access your deployed models under "Endpoints"

## Customization

To customize the workflow:

1. Replace the sample model training code with your own training script
2. Adjust the scoring script to match your model's requirements
3. Modify deployment configurations as needed

## Manual Execution

You can manually trigger the workflow from the GitHub Actions tab using the "workflow_dispatch" event.

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


