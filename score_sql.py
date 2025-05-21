import os
import json
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer


def init():
    """Initialize the model and tokenizer when the endpoint starts."""
    global model, tokenizer

    try:
        # Get the path where the model is stored
        model_dir = os.getenv("AZUREML_MODEL_DIR")
        logging.info(f"Loading model from {model_dir}")

        # Load model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)

        logging.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


def run(raw_data):
    """
    Generate SQL query from natural language input.

    Args:
        raw_data (str): JSON string containing the input text
        Format: {"input": "translate question to SQL: What is the total sales for 2023?"}

    Returns:
        dict: Contains generated SQL query
        Format: {"generated_sql": "SELECT SUM(sales) FROM sales_table WHERE YEAR(date) = 2023"}
    """
    try:
        # Parse input data
        data = json.loads(raw_data)
        input_text = data.get("input", "")

        if not input_text:
            return {"error": "No input text provided"}

        # Ensure input has the correct prefix if not already present
        if not input_text.lower().startswith("translate question to sql:"):
            input_text = "translate question to SQL: " + input_text

        # Tokenize input
        inputs = tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        )

        # Generate SQL query
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=False,
        )

        # Decode the generated SQL
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"generated_sql": generated_sql}

    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return {"error": str(e)}
