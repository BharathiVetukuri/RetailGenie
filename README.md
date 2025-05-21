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

## 🤖 Team Contributions

To be filled...

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- Python 3.9 or higher
- GCP project with Vertex AI + Artifact Registry enabled
- Google Cloud SDK installed + authenticated:

```bash
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
```
