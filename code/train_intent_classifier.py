import pandas as pd
import os
import argparse
import json
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

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
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_mapping)
)


def tokenize(example):
    return tokenizer(
        example["question"], truncation=True, padding="max_length", max_length=128
    )


dataset = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./results_intent_classifier",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_dir="./logs_intent",
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="no",
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# Save model, tokenizer, and label mapping to outputs directory for Azure ML
os.makedirs("outputs", exist_ok=True)
model.save_pretrained("outputs")
tokenizer.save_pretrained("outputs")
# Convert all values to int for JSON serialization
label_mapping_int = {k: int(v) for k, v in label_mapping.items()}
with open(os.path.join("outputs", "label_mapping.json"), "w") as f:
    json.dump(label_mapping_int, f)
print("✅ Model, tokenizer, and label mapping saved to outputs/")
