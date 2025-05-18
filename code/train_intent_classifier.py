import pandas as pd
import os
import argparse
import shutil
import tempfile
import json
from google.cloud import storage
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import torch

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# Load dataset
print("📦 Loading dataset from:", args.dataset_path)
df = pd.read_csv(args.dataset_path)
df = df[["question", "intent"]]

# Label encoding
le = LabelEncoder()
df["label"] = le.fit_transform(df["intent"])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
dataset = Dataset.from_pandas(df)

# Tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBERTTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

def tokenize(example):
    return tokenizer(example["question"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./results_intent_classifier",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_dir="./logs_intent",
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="no"
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# Save to temp dir
local_dir = tempfile.mkdtemp()
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

with open(os.path.join(local_dir, "label_mapping.json"), "w") as f:
    json.dump(label_mapping, f)

# Upload to GCS
gcs_model_path = os.path.join(args.output_dir, "intent")
bucket_name = gcs_model_path.split("/")[2]
base_path = "/".join(gcs_model_path.split("/")[3:])

client = storage.Client()

for fname in os.listdir(local_dir):
    local_path = os.path.join(local_dir, fname)
    gcs_blob_path = os.path.join(base_path, fname)

    print(f"⬆️ Uploading {fname} to gs://{bucket_name}/{gcs_blob_path}")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_path)

print(f"✅ Intent model successfully uploaded to gs://{bucket_name}/{base_path}")
