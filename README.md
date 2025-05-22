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
---
- Youtube Video Link : https://youtu.be/YXPZawbxc3s
- Report Link : https://docs.google.com/document/d/1DFvYuQQcZ2NcxfbIaWllHvJqroN-yr1INOVunanDdl8/edit?tab=t.0
- Slide Share : https://prezi.com/view/IsYkiUo37RbUuPnNzC5f/


## 🤖 Model Training & Datasets

For our SQL generation model, we fine-tuned the T5-small architecture on high-quality text-to-SQL datasets:

- [Clinton/Text-to-sql-v1](https://huggingface.co/datasets/Clinton/Text-to-sql-v1) - A comprehensive dataset with 262k examples of natural language questions paired with SQL queries
- [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) - Contains detailed SQL context with CREATE TABLE statements that helped our model understand database schemas

The combination of these datasets allowed our T5 model to effectively translate natural language retail questions into accurate SQL queries while understanding the underlying database structure. Our training approach focused on teaching the model to generate syntactically correct SQL while maintaining the semantic meaning of the original question.

## Screenshots

### Azure ML Pipeline

![Azure ML Pipeline](https://github.com/BharathiVetukuri/RetailGenie/blob/main/assets/images/AZURESS1.png)
_Screenshot shows our successfully deployed Azure ML pipeline with connected dataset and training steps_

### Full-Stack ML Application

![RetailGenie Dashboard](https://github.com/BharathiVetukuri/RetailGenie/blob/main/assets/images/SS1.png)
_Main dashboard of the RetailGenie application showing the query interface and visualization panel_

![Query Processing](https://github.com/BharathiVetukuri/RetailGenie/blob/main/assets/images/SS2.png)
_Natural language query being processed by our model with SQL generation_

![Data Visualization](https://github.com/BharathiVetukuri/RetailGenie/blob/main/assets/images/SS3.png)
_Interactive visualization of query results with charts and insights_

![Model Performance](https://github.com/BharathiVetukuri/RetailGenie/blob/main/assets/images/SS4.png)
_Performance metrics and model evaluation dashboard_

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
## Team Members and Contributions

**Soumya Bharathi Vetukuri**

- Key Contributions:

  - Designed the overall Vertex AI pipeline architecture for training and deployment

  - Wrote pipeline.py and integrated all pipeline steps (T5 training, Intent Classifier training, and deployment)

  - Implemented and tested submit_pipeline.py and compile_pipeline.py

  - Troubleshot MLOps errors including auth scopes, GCS pathing, and container failures

Validated success criteria for pipeline completion and GCS model outputs

**Shubham Jaysukhbhai Kothiya**

- Key Contributions:

- Preprocessed and curated the retail Q&A dataset

- Implemented train_sqlgen_t5.py using Hugging Face T5 for natural language to SQL generation

- Wrote train_intent_classifier.py using DistilBERT for intent classification

- Tuned model hyperparameters, added tokenization logic, and structured Hugging Face Trainer configs

- Managed local training experiments and transitioned models to cloud containers

**Rutuja Patil**

- Key Contributions:

- Designed Dockerfiles for all three containers: train-sqlgen, train-intent, and deploy

- Managed Artifact Registry and pushed all container images using docker build and docker push

- Debugged container runtime failures (e.g., missing packages like gcsfs, sentencepiece)

- Integrated GCS upload logic into the training scripts using google-cloud-storage

- Ensured containers authenticated properly inside Vertex AI jobs

**Yugm Patel**

- Key Contributions:

- Developed the local deploy_from_local.py script for deploying trained models to Vertex AI

- Handled aiplatform.Model.upload() and Endpoint.create() logic

- Set up environment-level authentication using ADC and verified GCP permissions

- Verified endpoint creation and tested response handling
