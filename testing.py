from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import json

# Path to your downloaded model artifacts
model_dir = "test_model"

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

# Load label mapping
with open(f"{model_dir}/label_mapping.json") as f:
    label_mapping = json.load(f)
# Invert the mapping for prediction
inv_label_mapping = {str(v): k for k, v in label_mapping.items()}

# Test input
input_text = "Show me the sales report for last month"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
predicted_intent = inv_label_mapping[str(pred)]

print("Predicted intent:", predicted_intent)
