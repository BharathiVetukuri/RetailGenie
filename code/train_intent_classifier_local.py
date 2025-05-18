import pandas as pd
import os
import argparse
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# Get project root directory (the directory containing the script's parent directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "retail_dataset.csv"),
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.path.join(PROJECT_ROOT, "model_intent_classifier"),
)
args = parser.parse_args()

print("📦 Loading dataset from:", args.dataset_path)
df = pd.read_csv(args.dataset_path)
df = df[["question", "intent"]]

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["intent"])
label_mapping = {intent: idx for idx, intent in enumerate(label_encoder.classes_)}

# Create dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
num_labels = len(label_mapping)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)


def preprocess(example):
    return tokenizer(
        example["question"], truncation=True, padding="max_length", max_length=128
    )


tokenized_dataset = dataset.map(preprocess)

# Create output directories relative to project root
logs_dir = os.path.join(PROJECT_ROOT, "logs_intent")
results_dir = os.path.join(PROJECT_ROOT, "results_intent")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=results_dir,
    per_device_train_batch_size=8,
    num_train_epochs=3,  # Reduced for faster training
    logging_dir=logs_dir,
    logging_steps=5,
    save_strategy="epoch",
    # evaluation_strategy="no",
    eval_strategy="no",
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("🚀 Starting training...")
trainer.train()

# Save model locally
print(f"💾 Saving model to {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# Save label mapping
with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
    json.dump(label_mapping, f)

print(f"✅ Model successfully saved to {args.output_dir}")
