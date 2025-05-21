import json
import os
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch


def init():
    global model, tokenizer, label_mapping, inv_label_mapping
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    with open(os.path.join(model_dir, "label_mapping.json")) as f:
        label_mapping = json.load(f)
    inv_label_mapping = {str(v): k for k, v in label_mapping.items()}


def run(raw_data):
    data = json.loads(raw_data)
    text = data["input"]
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    predicted_intent = inv_label_mapping[str(pred)]
    return {"intent": predicted_intent}
