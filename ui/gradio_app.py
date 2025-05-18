import gradio as gr
import pandas as pd
import sqlite3
import torch
import os
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)
import json
import matplotlib.pyplot as plt
import re

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set model paths
sql_model_path = os.path.join(PROJECT_ROOT, "model_sqlgen_t5")
intent_model_path = os.path.join(PROJECT_ROOT, "model_intent_classifier")
chart_path = os.path.join(PROJECT_ROOT, "chart.png")

# Load T5 SQL generator
t5_model = T5ForConditionalGeneration.from_pretrained(sql_model_path)
t5_tokenizer = T5Tokenizer.from_pretrained(sql_model_path)

# Load intent classifier
intent_model = DistilBertForSequenceClassification.from_pretrained(intent_model_path)
intent_tokenizer = DistilBertTokenizerFast.from_pretrained(intent_model_path)
with open(os.path.join(intent_model_path, "label_mapping.json")) as f:
    label_mapping = json.load(f)
id2label = {v: k for k, v in label_mapping.items()}

# Enhanced sample data for the in-memory database
retail_data = {
    "product_name": [
        "Sneakers",
        "T-Shirt",
        "Laptop",
        "Sneakers",
        "T-Shirt",
        "Jeans",
        "Smartwatch",
        "Hoodie",
        "Tablet",
        "Backpack",
    ],
    "category": [
        "Footwear",
        "Apparel",
        "Electronics",
        "Footwear",
        "Apparel",
        "Apparel",
        "Electronics",
        "Apparel",
        "Electronics",
        "Accessories",
    ],
    "quantity": [5, 3, 2, 7, 2, 3, 1, 2, 1, 4],
    "total_price": [500, 90, 2000, 700, 60, 240, 350, 100, 800, 150],
    "store_id": [1, 1, 2, 2, 1, 3, 2, 1, 3, 1],
    "payment_method": [
        "Credit Card",
        "Cash",
        "Credit Card",
        "Debit Card",
        "Cash",
        "Credit Card",
        "Mobile Payment",
        "Credit Card",
        "Debit Card",
        "Cash",
    ],
    "date": pd.to_datetime(
        [
            "2023-03-01",
            "2023-03-02",
            "2023-04-01",
            "2023-03-15",
            "2023-03-20",
            "2023-03-05",
            "2023-03-10",
            "2023-03-22",
            "2023-04-05",
            "2023-04-10",
        ]
    ), 
}
sample_df = pd.DataFrame(retail_data)

# Default SQL queries for common questions
default_queries = {
    "top selling products in march": "SELECT product_name, SUM(quantity) AS total_quantity FROM transactions WHERE strftime('%m', date) = '03' GROUP BY product_name ORDER BY total_quantity DESC LIMIT 5;",
    "total sales for each category": "SELECT category, SUM(total_price) AS total_sales FROM transactions GROUP BY category ORDER BY total_sales DESC;",
    "sales in march": "SELECT product_name, SUM(quantity) AS quantity_sold FROM transactions WHERE strftime('%m', date) = '03' GROUP BY product_name;",
    "compare sales between categories": "SELECT category, SUM(total_price) AS total_sales FROM transactions GROUP BY category;",
    "which store has highest revenue": "SELECT store_id, SUM(total_price) AS total_revenue FROM transactions GROUP BY store_id ORDER BY total_revenue DESC LIMIT 1;",
    "popular payment methods": "SELECT payment_method, COUNT(*) AS frequency FROM transactions GROUP BY payment_method ORDER BY frequency DESC;",
    "bestselling products overall": "SELECT product_name, SUM(quantity) AS total_sold FROM transactions GROUP BY product_name ORDER BY total_sold DESC LIMIT 5;",
}


# Generate insights from query results
def generate_insights(result_df, intent, question):
    insights = ""

    if result_df is None or result_df.empty:
        return "No data available for insights."

    try:
        # Basic insights
        if "top" in question.lower() and len(result_df) > 0:
            top_item = result_df.iloc[0]
            insights += f"• {top_item.iloc[0]} is the top performer with {top_item.iloc[1]} units.\n"

        if ("compare" in question.lower() or "comparison" in intent) and len(
            result_df
        ) >= 2:
            highest = result_df.iloc[0]
            lowest = result_df.iloc[-1]
            insights += f"• {highest.iloc[0]} has {round((highest.iloc[1]/lowest.iloc[1]-1)*100, 1)}% higher performance than {lowest.iloc[0]}.\n"

        # Category insights
        if "category" in result_df.columns:
            categories = result_df["category"].nunique()
            insights += f"• Analysis covers {categories} different categories.\n"

        # Add a recommendation if we have enough data
        if len(result_df) > 2:
            insights += (
                "• Recommendation: Consider further analysis by store location.\n"
            )

    except Exception as e:
        insights += f"Error generating insights: {str(e)}\n"

    return insights


# Check if SQL is valid
def is_valid_sql(sql_query):
    # Check if it starts with SELECT and contains basic SQL elements
    return re.search(r"^SELECT\s+.+\s+FROM\s+.+", sql_query, re.IGNORECASE) is not None


# Inference function
def retailgenie_pipeline(question, chart_type="auto"):
    # Create a new connection for this request to avoid thread issues
    conn = sqlite3.connect(":memory:")

    # Load data into the new connection
    sample_df.to_sql("transactions", conn, index=False, if_exists="replace")

    # Intent classification
    inputs = intent_tokenizer(
        question, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = intent_model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        intent = id2label[predicted_class_id]

    # SQL generation
    prompt = "translate question to SQL: " + question
    input_ids = t5_tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output_ids = t5_model.generate(input_ids, max_length=128)
    sql = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Check if the generated SQL is valid, if not use default queries
    if not is_valid_sql(sql):
        # Try to find a matching default query
        for key, default_sql in default_queries.items():
            if key in question.lower():
                sql = default_sql
                break
        else:
            # If no match found, use a simple fallback query
            sql = "SELECT product_name, SUM(quantity) as total_quantity FROM transactions GROUP BY product_name ORDER BY total_quantity DESC;"

    print(f"Executing SQL: {sql}")

    # Execute SQL
    try:
        result_df = pd.read_sql_query(sql, conn)

        # Generate insights
        insights = generate_insights(result_df, intent, question)

        # Plot chart if we have data
        if not result_df.empty:
            # Set dark theme for the chart
            plt.style.use("dark_background")
            plt.figure(figsize=(8, 5))

            # Use user-selected chart type or default based on intent
            if chart_type == "bar" or (
                chart_type == "auto" and (intent == "summary" or intent == "comparison")
            ):
                ax = result_df.plot(
                    kind="bar",
                    x=result_df.columns[0],
                    y=result_df.columns[1],
                    legend=False,
                    color="#1e88e5",
                )
            elif chart_type == "line" or (chart_type == "auto" and intent == "trend"):
                ax = result_df.plot(
                    kind="line",
                    x=result_df.columns[0],
                    y=result_df.columns[1],
                    legend=False,
                    marker="o",
                    color="#1e88e5",
                )
            elif chart_type == "pie":
                ax = result_df.plot(
                    kind="pie",
                    y=result_df.columns[1],
                    labels=result_df[result_df.columns[0]],
                    autopct="%1.1f%%",
                )
                plt.ylabel("")  # Hide y-label for pie chart
            elif chart_type == "horizontal" or (
                chart_type == "auto" and intent == "anomaly"
            ):
                ax = result_df.plot(
                    kind="barh",
                    x=result_df.columns[0],
                    y=result_df.columns[1],
                    legend=False,
                    color="#1e88e5",
                )
            else:
                ax = result_df.plot(kind="bar", legend=False, color="#1e88e5")

            # Enhance chart appearance
            plt.title(question, fontsize=14)
            plt.xlabel(result_df.columns[0], fontsize=12)
            if chart_type != "pie":
                plt.ylabel(result_df.columns[1], fontsize=12)

            plt.grid(axis="y", linestyle="--", alpha=0.7, color="#444444")
            plt.tight_layout()

            # Rotate x-axis labels for better readability
            if chart_type != "pie" and chart_type != "horizontal":
                plt.xticks(rotation=45, ha="right")

            # Add data labels on bars
            if chart_type in ["bar", "auto"] and len(result_df) <= 10:
                for i, v in enumerate(result_df[result_df.columns[1]]):
                    ax.text(i, v + 0.1, str(round(v, 1)), ha="center", color="white")

            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            chart = chart_path
        else:
            chart = None
    except Exception as e:
        error_message = f"Error executing SQL: {e}"
        print(error_message)
        # Close the connection before returning
        conn.close()
        return None, intent, sql, None, "Error executing query"

    # Close the connection before returning
    conn.close()
    return result_df, intent, sql, chart_path, insights


# Custom CSS for dark theme
css = """
body {
    background-color: #1e1e1e !important;
    color: white !important;
}
.gradio-container {
    background-color: #1e1e1e !important;
}
.input-box, .output-box {
    background-color: #2d2d2d !important;
    border: 1px solid #424242 !important;
}
button.primary {
    background-color: #ff6b1a !important;
    color: white !important;
}
button.secondary {
    background-color: #424242 !important;
    color: white !important;
}
table {
    background-color: #2d2d2d !important;
    color: white !important;
}
th, td {
    border: 1px solid #424242 !important;
}
.footer {
    margin-top: 20px;
    text-align: center;
    font-size: 12px;
    color: #d9d9e3;
}
"""

# Example questions to help users
example_questions = [
    "What are the top 5 selling products in March?",
    "Show me total sales for each category",
    "Which store has the highest revenue?",
    "What is the most popular payment method?",
    "Compare sales between different categories",
]

# Create a simplified Gradio interface with Blocks for more flexibility
with gr.Blocks(css=css, title="RetailGenie - Advanced Retail Analytics") as demo:
    gr.Markdown(
        """# RetailGenie – An LLM-Powered BI Assistant
        Ask retail business questions and get SQL + chart answers powered by fine-tuned T5 and BERT models."""
    )

    with gr.Row():
        with gr.Column(scale=3):
            # Input area
            question_input = gr.Textbox(
                label="Ask a retail-related question",
                placeholder="e.g., What are the top 5 selling products in March?",
                lines=3,
            )

            # Example questions
            gr.Examples(examples=example_questions, inputs=question_input)

            # Chart type selector
            chart_type = gr.Radio(
                ["auto", "bar", "line", "pie", "horizontal"],
                label="Chart Type",
                value="auto",
            )

            # Buttons
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")

            # Output area - SQL and Intent
            with gr.Accordion("SQL and Intent Details", open=False):
                intent_output = gr.Textbox(label="Predicted Intent")
                sql_output = gr.Textbox(label="Generated SQL", lines=3)

        with gr.Column(scale=4):
            # Results
            results_df = gr.DataFrame(label="Query Result")
            chart_output = gr.Image(label="Chart")
            insights_output = gr.Textbox(label="Insights", lines=5)

    # Footer
    gr.Markdown("Built with Gradio • RetailGenie v1.5", elem_classes="footer")

    # Set up interactivity
    submit_btn.click(
        fn=retailgenie_pipeline,
        inputs=[question_input, chart_type],
        outputs=[results_df, intent_output, sql_output, chart_output, insights_output],
    )

    clear_btn.click(
        fn=lambda: ["", "auto", None, "", "", None, ""],
        inputs=[],
        outputs=[
            question_input,
            chart_type,
            results_df,
            intent_output,
            sql_output,
            chart_output,
            insights_output,
        ],
    )

if __name__ == "__main__":
    demo.launch()