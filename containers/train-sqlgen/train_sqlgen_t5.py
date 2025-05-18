import pandas as pd
import os
import argparse
import shutil
import tempfile
from google.cloud import storage
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import torch

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

print("📦 Loading dataset from:", args.dataset_path)
df = pd.read_csv(args.dataset_path)
df = df[["question", "sql"]].rename(columns={"question": "input_text", "sql": "target_text"})
df["input_text"] = "translate question to SQL: " + df["input_text"]
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess(example):
    input_enc = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=128)
    target_enc = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

tokenized_dataset = dataset.map(preprocess)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_t5_sqlgen",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="no"
)

# Train model
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()

# Save model to temporary local directory
local_dir = tempfile.mkdtemp()
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

# Upload all files to GCS
gcs_model_path = os.path.join(args.output_dir, "sqlgen")
bucket_name = gcs_model_path.split("/")[2]
base_path = "/".join(gcs_model_path.split("/")[3:])

client = storage.Client()

for fname in os.listdir(local_dir):
    local_path = os.path.join(local_dir, fname)
    gcs_blob_path = os.path.join(base_path, fname)

    print(f"⬆️  Uploading {fname} to gs://{bucket_name}/{gcs_blob_path}")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_path)

print(f"✅ Model successfully uploaded to gs://{bucket_name}/{base_path}")
