import pandas as pd
import os
import argparse
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

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
    "--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "model_sqlgen_t5")
)
args = parser.parse_args()

print("📦 Loading dataset from:", args.dataset_path)
df = pd.read_csv(args.dataset_path)
df = df[["question", "sql"]].rename(
    columns={"question": "input_text", "sql": "target_text"}
)
df["input_text"] = "translate question to SQL: " + df["input_text"]
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def preprocess(example):
    input_enc = tokenizer(
        example["input_text"], truncation=True, padding="max_length", max_length=128
    )
    target_enc = tokenizer(
        example["target_text"], truncation=True, padding="max_length", max_length=128
    )
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc


tokenized_dataset = dataset.map(preprocess)

# Create output directories relative to project root
logs_dir = os.path.join(PROJECT_ROOT, "logs")
results_dir = os.path.join(PROJECT_ROOT, "results_t5_sqlgen")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=results_dir,
    per_device_train_batch_size=4,
    num_train_epochs=3,  # Reduced for faster training
    logging_dir=logs_dir,
    logging_steps=5,
    save_strategy="epoch",
)

# Train model
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
print("🚀 Starting training...")
trainer.train()

# Save model locally
print(f"💾 Saving model to {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"✅ Model successfully saved to {args.output_dir}")
