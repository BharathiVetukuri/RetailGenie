import json
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging


def init():
    """Initialize model. This function is called when the container is started."""
    global model, tokenizer

    logging.info("Initializing model...")

    # Get model directory
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "model")

    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logging.info("Model loaded successfully")


def run(raw_data):
    """Inference function. This function is called for every request."""
    try:
        # Parse input
        data = json.loads(raw_data)
        inputs = data.get("inputs", [])

        # Set default parameters
        max_length = data.get("max_length", 128)
        num_return_sequences = data.get("num_return_sequences", 1)

        # Process inputs
        results = []
        for input_text in inputs:
            input_ids = tokenizer.encode(input_text, return_tensors="pt")

            # Move to same device as model
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)

            # Generate output
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

            # Decode output
            decoded_outputs = []
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                decoded_outputs.append(decoded)

            results.append(decoded_outputs)

        # Return results
        return json.dumps({"predictions": results})

    except Exception as e:
        error = str(e)
        logging.error(f"Error during inference: {error}")
        return json.dumps({"error": error})
