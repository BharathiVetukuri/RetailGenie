import os
import json
import logging
from typing import Dict, Any
import azure.functions as func
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
import torch
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import io
import base64

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetailGenieModel:
    def __init__(self):
        # Load models from Azure ML model registry
        self.sql_model_path = os.getenv("SQL_MODEL_PATH", "model_sqlgen_t5")
        self.intent_model_path = os.getenv(
            "INTENT_MODEL_PATH", "model_intent_classifier"
        )

        # Initialize models
        self.sql_model = T5ForConditionalGeneration.from_pretrained(self.sql_model_path)
        self.sql_tokenizer = T5Tokenizer.from_pretrained(self.sql_model_path)

        self.intent_model = DistilBertForSequenceClassification.from_pretrained(
            self.intent_model_path
        )
        self.intent_tokenizer = DistilBertTokenizer.from_pretrained(
            self.intent_model_path
        )

        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sql_model.to(self.device)
        self.intent_model.to(self.device)

        # Load dataset
        self.df = pd.read_csv("data/testing_sql_data.csv")

        # Create SQLite connection
        self.conn = sqlite3.connect(":memory:")
        self.df.to_sql("retail_data", self.conn, index=False)

        logger.info("Models and data loaded successfully")

    def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language question."""
        inputs = self.sql_tokenizer.encode(
            f"translate English to SQL: {question}",
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        outputs = self.sql_model.generate(
            inputs, max_length=128, num_beams=4, early_stopping=True
        )

        sql_query = self.sql_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return sql_query

    def classify_intent(self, question: str) -> str:
        """Classify the intent of the question."""
        inputs = self.intent_tokenizer(
            question, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        outputs = self.intent_model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        intent_idx = torch.argmax(predictions).item()

        intents = ["trend", "comparison", "aggregation", "filtering"]
        return intents[intent_idx]

    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query on the dataset."""
        try:
            return pd.read_sql_query(sql_query, self.conn)
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            return pd.DataFrame()

    def generate_chart(self, df: pd.DataFrame, intent: str) -> str:
        """Generate chart based on the query results and intent."""
        plt.figure(figsize=(10, 6))

        if intent == "trend":
            plt.plot(df.iloc[:, 0], df.iloc[:, 1])
            plt.title("Trend Analysis")
        elif intent == "comparison":
            df.plot(kind="bar", x=df.columns[0], y=df.columns[1])
            plt.title("Comparison Analysis")
        elif intent == "aggregation":
            df.plot(kind="pie", y=df.columns[1], labels=df[df.columns[0]])
            plt.title("Aggregation Analysis")
        else:
            df.plot(kind="bar", x=df.columns[0], y=df.columns[1])
            plt.title("Data Analysis")

        plt.tight_layout()

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()

        return img_str

    def process_query(self, question: str) -> Dict[str, Any]:
        """Process a natural language query and return results."""
        try:
            # Generate SQL
            sql_query = self.generate_sql(question)

            # Classify intent
            intent = self.classify_intent(question)

            # Execute SQL
            result_df = self.execute_sql(sql_query)

            # Generate chart
            chart_base64 = self.generate_chart(result_df, intent)

            return {
                "sql_query": sql_query,
                "intent": intent,
                "results": result_df.to_dict(orient="records"),
                "chart": chart_base64,
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "error": str(e),
                "sql_query": "",
                "intent": "",
                "results": [],
                "chart": "",
            }


# Initialize the model
model = RetailGenieModel()


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function entry point."""
    try:
        # Get the question from the request
        req_body = req.get_json()
        question = req_body.get("question")

        if not question:
            return func.HttpResponse(
                "Please provide a question in the request body", status_code=400
            )

        # Process the query
        result = model.process_query(question)

        return func.HttpResponse(
            json.dumps(result), mimetype="application/json", status_code=200
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}), mimetype="application/json", status_code=500
        )
