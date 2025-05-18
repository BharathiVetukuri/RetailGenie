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
from datetime import datetime, timedelta
import io
import base64

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
    "transaction_id": range(1, 21),
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
        "Sneakers",
        "T-Shirt",
        "Laptop",
        "Smartphone",
        "Bluetooth Headphones",
        "Jeans",
        "Sneakers",
        "Wireless Earbuds",
        "Smart Speaker",
        "External SSD",
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
        "Footwear",
        "Apparel",
        "Electronics",
        "Electronics",
        "Electronics",
        "Apparel",
        "Footwear",
        "Electronics",
        "Electronics",
        "Electronics",
    ],
    "quantity": [5, 3, 2, 7, 2, 3, 1, 2, 1, 4, 4, 2, 1, 2, 3, 2, 3, 2, 1, 1],
    "total_price": [
        500,
        90,
        2000,
        700,
        60,
        240,
        350,
        100,
        800,
        150,
        400,
        60,
        1800,
        1500,
        200,
        160,
        300,
        180,
        120,
        90,
    ],
    "unit_price": [
        100,
        30,
        1000,
        100,
        30,
        80,
        350,
        50,
        800,
        37.5,
        100,
        30,
        1800,
        750,
        66.67,
        80,
        100,
        90,
        120,
        90,
    ],
    "store_id": [1, 1, 2, 2, 1, 3, 2, 1, 3, 1, 2, 3, 1, 2, 3, 1, 3, 2, 1, 3],
    "customer_id": [
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
    ],
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
        "Credit Card",
        "Mobile Payment",
        "Credit Card",
        "Debit Card",
        "Cash",
        "Mobile Payment",
        "Credit Card",
        "Debit Card",
        "Mobile Payment",
        "Credit Card",
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
            "2023-02-15",
            "2023-02-20",
            "2023-02-25",
            "2023-03-08",
            "2023-03-12",
            "2023-03-18",
            "2023-03-25",
            "2023-04-02",
            "2023-04-08",
            "2023-04-15",
        ]
    ),
}
sample_df = pd.DataFrame(retail_data)

# Store information
stores = {
    1: {"name": "Downtown Store", "location": "123 Main St", "manager": "John Smith"},
    2: {"name": "Shopping Mall", "location": "456 Market Ave", "manager": "Jane Doe"},
    3: {
        "name": "Outlet Center",
        "location": "789 Commerce Blvd",
        "manager": "Robert Johnson",
    },
}

# Default SQL queries for common questions
default_queries = {
    "top selling products in march": "SELECT product_name, SUM(quantity) AS total_quantity FROM transactions WHERE strftime('%m', date) = '03' GROUP BY product_name ORDER BY total_quantity DESC LIMIT 5;",
    "total sales for each category": "SELECT category, SUM(total_price) AS total_sales FROM transactions GROUP BY category ORDER BY total_sales DESC;",
    "sales in march": "SELECT product_name, SUM(quantity) AS quantity_sold FROM transactions WHERE strftime('%m', date) = '03' GROUP BY product_name;",
    "compare sales between categories": "SELECT category, SUM(total_price) AS total_sales FROM transactions GROUP BY category;",
    "which store has highest revenue": "SELECT store_id, SUM(total_price) AS total_revenue FROM transactions GROUP BY store_id ORDER BY total_revenue DESC LIMIT 1;",
    "average transaction value": "SELECT AVG(total_price) AS avg_transaction_value FROM transactions;",
    "popular payment methods": "SELECT payment_method, COUNT(*) AS frequency FROM transactions GROUP BY payment_method ORDER BY frequency DESC;",
    "bestselling products overall": "SELECT product_name, SUM(quantity) AS total_sold FROM transactions GROUP BY product_name ORDER BY total_sold DESC LIMIT 5;",
    "product performance by store": "SELECT store_id, product_name, SUM(quantity) AS total_sold FROM transactions GROUP BY store_id, product_name ORDER BY store_id, total_sold DESC;",
    "daily sales trend": "SELECT date, SUM(total_price) AS daily_sales FROM transactions GROUP BY date ORDER BY date;",
    "customer spending": "SELECT customer_id, SUM(total_price) AS total_spent FROM transactions GROUP BY customer_id ORDER BY total_spent DESC;",
}


# Insights generation functions
def generate_key_insights(result_df, intent, question, sql):
    insights = []

    if result_df is None or result_df.empty:
        return ["No data available to generate insights."]

    try:
        # Basic insights based on intent and data
        if "top" in question.lower() and len(result_df) > 0:
            top_item = result_df.iloc[0]
            insights.append(
                f"✓ {top_item.iloc[0]} is the top performer with {top_item.iloc[1]} units."
            )

        if "compare" in question.lower() or "comparison" in intent:
            if len(result_df) >= 2:
                highest = result_df.iloc[0]
                lowest = result_df.iloc[-1]
                insights.append(
                    f"✓ {highest.iloc[0]} has {round((highest.iloc[1]/lowest.iloc[1]-1)*100, 1)}% higher performance than {lowest.iloc[0]}."
                )

        # Detect trends or patterns
        if (
            "total_price" in result_df.columns
            or "total_sales" in result_df.columns
            or "revenue" in result_df.columns
        ):
            value_col = next(
                (
                    col
                    for col in [
                        "total_price",
                        "total_sales",
                        "revenue",
                        "total_revenue",
                    ]
                    if col in result_df.columns
                ),
                result_df.columns[1],
            )
            total_value = result_df[value_col].sum()
            insights.append(f"✓ Total value: ${total_value:,.2f}")

            if len(result_df) > 1:
                avg_value = result_df[value_col].mean()
                insights.append(f"✓ Average value: ${avg_value:,.2f}")

        # Product specific insights
        if "product_name" in result_df.columns and "quantity" in result_df.columns:
            total_quantity = result_df["quantity"].sum()
            insights.append(f"✓ Total quantity sold: {total_quantity} units")

        # Time-based insights
        if "date" in result_df.columns:
            if len(result_df) > 1:
                date_col = result_df["date"]
                date_range = f"{date_col.min().strftime('%Y-%m-%d')} to {date_col.max().strftime('%Y-%m-%d')}"
                insights.append(f"✓ Date range: {date_range}")

        # Category insights
        if "category" in result_df.columns:
            categories = result_df["category"].nunique()
            insights.append(f"✓ Analysis covers {categories} different categories.")

        # Store insights
        if "store_id" in result_df.columns:
            store_count = result_df["store_id"].nunique()
            insights.append(f"✓ Data from {store_count} different stores.")

        # Add a recommendation if we have enough data
        if len(result_df) > 2:
            insights.append(
                "💡 Recommendation: Consider further analysis by store location to identify regional patterns."
            )

    except Exception as e:
        insights.append(f"Error generating insights: {str(e)}")

    return insights


# Check if SQL is valid
def is_valid_sql(sql_query):
    # Check if it starts with SELECT and contains basic SQL elements
    return re.search(r"^SELECT\s+.+\s+FROM\s+.+", sql_query, re.IGNORECASE) is not None


# Data export function
def export_to_csv(df):
    if df is None or df.empty:
        return None

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    csv_bytes = csv_str.encode()
    b64 = base64.b64encode(csv_bytes).decode()
    href = f"data:text/csv;base64,{b64}"
    return href


# Calculate key metrics
def calculate_metrics(conn):
    metrics = {}

    try:
        # Total revenue
        total_revenue = pd.read_sql_query(
            "SELECT SUM(total_price) AS value FROM transactions", conn
        ).iloc[0, 0]
        metrics["total_revenue"] = f"${total_revenue:,.2f}"

        # Total transactions
        total_transactions = pd.read_sql_query(
            "SELECT COUNT(*) AS value FROM transactions", conn
        ).iloc[0, 0]
        metrics["total_transactions"] = f"{total_transactions}"

        # Average transaction value
        avg_value = pd.read_sql_query(
            "SELECT AVG(total_price) AS value FROM transactions", conn
        ).iloc[0, 0]
        metrics["avg_value"] = f"${avg_value:,.2f}"

        # Top category
        top_category = pd.read_sql_query(
            "SELECT category, SUM(total_price) AS total FROM transactions GROUP BY category ORDER BY total DESC LIMIT 1",
            conn,
        )
        if not top_category.empty:
            metrics["top_category"] = top_category.iloc[0, 0]
        else:
            metrics["top_category"] = "N/A"

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        for key in ["total_revenue", "total_transactions", "avg_value", "top_category"]:
            if key not in metrics:
                metrics[key] = "Error"

    return metrics


# Inference function
def retailgenie_pipeline(
    question, date_range=None, store_filter=None, chart_type=None, theme="dark"
):
    # Create a new connection for this request to avoid thread issues
    conn = sqlite3.connect(":memory:")

    # Copy of the data for filtering
    filtered_df = sample_df.copy()

    # Apply date filter if specified
    if date_range and date_range != "all":
        today = datetime.now().date()
        if date_range == "last_7_days":
            start_date = today - timedelta(days=7)
            filtered_df = filtered_df[filtered_df["date"] >= pd.Timestamp(start_date)]
        elif date_range == "last_30_days":
            start_date = today - timedelta(days=30)
            filtered_df = filtered_df[filtered_df["date"] >= pd.Timestamp(start_date)]
        elif date_range == "last_90_days":
            start_date = today - timedelta(days=90)
            filtered_df = filtered_df[filtered_df["date"] >= pd.Timestamp(start_date)]
        elif date_range == "this_month":
            start_date = datetime(today.year, today.month, 1)
            filtered_df = filtered_df[filtered_df["date"] >= pd.Timestamp(start_date)]
        elif date_range == "this_quarter":
            quarter_month = (today.month - 1) // 3 * 3 + 1
            start_date = datetime(today.year, quarter_month, 1)
            filtered_df = filtered_df[filtered_df["date"] >= pd.Timestamp(start_date)]

    # Apply store filter if specified
    if store_filter and store_filter != "all":
        store_id = int(store_filter)
        filtered_df = filtered_df[filtered_df["store_id"] == store_id]

    # Load filtered data into the new connection
    filtered_df.to_sql("transactions", conn, index=False, if_exists="replace")

    # Calculate metrics based on filtered data
    metrics = calculate_metrics(conn)

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
        insights = generate_key_insights(result_df, intent, question, sql)

        # Generate CSV export link
        export_link = export_to_csv(result_df)

        # Plot chart if we have data
        if not result_df.empty:
            # Set the chart style based on the theme
            if theme == "dark":
                plt.style.use("dark_background")
                text_color = "white"
                grid_color = "#444444"
                edge_color = "#666666"
                bar_color = "#1e88e5"
            else:
                plt.style.use("default")
                text_color = "black"
                grid_color = "#cccccc"
                edge_color = "#dddddd"
                bar_color = "#1e88e5"

            plt.figure(figsize=(8, 5))

            # Use the specified chart type or default based on intent
            plot_type = chart_type if chart_type else "auto"

            if plot_type == "bar" or (
                plot_type == "auto" and (intent == "summary" or intent == "comparison")
            ):
                ax = result_df.plot(
                    kind="bar",
                    x=result_df.columns[0],
                    y=result_df.columns[1],
                    legend=False,
                    color=bar_color,
                    edgecolor=edge_color,
                )
            elif plot_type == "line" or (plot_type == "auto" and intent == "trend"):
                ax = result_df.plot(
                    kind="line",
                    x=result_df.columns[0],
                    y=result_df.columns[1],
                    legend=False,
                    marker="o",
                    color=bar_color,
                )
            elif plot_type == "pie":
                ax = result_df.plot(
                    kind="pie",
                    y=result_df.columns[1],
                    labels=result_df[result_df.columns[0]],
                    autopct="%1.1f%%",
                )
                plt.ylabel("")  # Hide y-label for pie chart
            elif plot_type == "horizontal" or (
                plot_type == "auto" and intent == "anomaly"
            ):
                ax = result_df.plot(
                    kind="barh",
                    x=result_df.columns[0],
                    y=result_df.columns[1],
                    legend=False,
                    color=bar_color,
                    edgecolor=edge_color,
                )
            elif plot_type == "scatter":
                # Only use scatter if we have at least two numeric columns
                numeric_cols = result_df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) >= 2:
                    ax = result_df.plot(
                        kind="scatter",
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        legend=False,
                        color=bar_color,
                    )
                else:
                    ax = result_df.plot(
                        kind="bar", legend=False, color=bar_color, edgecolor=edge_color
                    )
            else:
                ax = result_df.plot(
                    kind="bar", legend=False, color=bar_color, edgecolor=edge_color
                )

            # Enhance chart appearance
            plt.title(question, color=text_color, fontsize=14)
            plt.xlabel(result_df.columns[0], color=text_color, fontsize=12)
            if plot_type != "pie":
                plt.ylabel(result_df.columns[1], color=text_color, fontsize=12)

            plt.grid(axis="y", linestyle="--", alpha=0.7, color=grid_color)
            plt.xticks(color=text_color)
            plt.yticks(color=text_color)
            plt.tight_layout()

            # Rotate x-axis labels for better readability
            if plot_type != "pie" and plot_type != "horizontal":
                plt.xticks(rotation=45, ha="right")

            # Add data labels on bars
            if plot_type in ["bar", "auto"] and len(result_df) <= 10:
                for i, v in enumerate(result_df[result_df.columns[1]]):
                    ax.text(i, v + 0.1, str(round(v, 1)), ha="center", color=text_color)

            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            chart = chart_path
        else:
            chart = None
    except Exception as e:
        error_message = f"Error executing SQL: {e}"
        print(error_message)
        # Close the connection before returning
        conn.close()
        return None, None, intent, sql, None, None, metrics, None

    # Close the connection before returning
    conn.close()
    return (
        result_df,
        export_link,
        intent,
        sql,
        chart_path,
        insights,
        metrics,
        filtered_df,
    )


# Function to switch to results tab after query
def switch_to_results_tab():
    return 1


# Simple CSS to improve the layout and create a dark theme
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
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: white;
}
.metric-label {
    font-size: 16px;
    color: #d9d9e3;
}
.footer {
    margin-top: 20px;
    text-align: center;
    font-size: 12px;
    color: #d9d9e3;
}
.top-header {
    margin-bottom: 0;
    border-bottom: 1px solid #424242;
    padding-bottom: 10px;
}
.tabs {
    border-bottom: 1px solid #424242 !important;
}
.tab-selected {
    border-bottom: 2px solid #ff6b1a !important;
}
.panel {
    background-color: #1e1e1e !important;
}
table {
    background-color: #2d2d2d !important;
    color: white !important;
}
th, td {
    border: 1px solid #424242 !important;
}
.insight-card {
    background-color: #2d2d2d;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    border-left: 4px solid #ff6b1a;
}
.metric-card {
    background-color: #2d2d2d;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.theme-toggle {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
}
.download-button {
    margin-top: 10px;
}
.filter-container {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
}
.visualization-controls {
    display: flex;
    gap: 10px;
    margin: 10px 0;
    padding: 10px;
    background-color: #2d2d2d;
    border-radius: 8px;
}
"""

# Light theme CSS
light_css = """
body {
    background-color: #f8f9fa !important;
    color: #333333 !important;
}
.gradio-container {
    background-color: #f8f9fa !important;
}
.input-box, .output-box {
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
}
button.primary {
    background-color: #ff6b1a !important;
    color: white !important;
}
button.secondary {
    background-color: #e0e0e0 !important;
    color: #333333 !important;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #333333;
}
.metric-label {
    font-size: 16px;
    color: #666666;
}
.footer {
    margin-top: 20px;
    text-align: center;
    font-size: 12px;
    color: #666666;
}
.top-header {
    margin-bottom: 0;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 10px;
}
.tabs {
    border-bottom: 1px solid #e0e0e0 !important;
}
.tab-selected {
    border-bottom: 2px solid #ff6b1a !important;
}
.panel {
    background-color: #ffffff !important;
}
table {
    background-color: #ffffff !important;
    color: #333333 !important;
}
th, td {
    border: 1px solid #e0e0e0 !important;
}
.insight-card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    border-left: 4px solid #ff6b1a;
}
.metric-card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
"""


# Function to switch themes
def toggle_theme(theme_state):
    return "light" if theme_state == "dark" else "dark"


# Create a simplified Gradio interface
with gr.Blocks(css=css, title="RetailGenie - Retail Analytics Dashboard") as demo:
    # State variables
    theme_state = gr.State("dark")
    current_results = gr.State(None)
    current_df = gr.State(None)

    # Header with title
    with gr.Row(elem_classes="top-header"):
        gr.Markdown(
            """# RetailGenie - Advanced Retail Analytics Dashboard
            Ask retail business questions and get SQL + chart answers powered by fine-tuned T5 and BERT models."""
        )

    # Filters row
    with gr.Row(elem_classes="filter-container"):
        date_filter = gr.Dropdown(
            choices=[
                "all",
                "last_7_days",
                "last_30_days",
                "last_90_days",
                "this_month",
                "this_quarter",
            ],
            value="all",
            label="Time Period",
            elem_classes="filter",
        )
        store_filter = gr.Dropdown(
            choices=["all", "1", "2", "3"],
            value="all",
            label="Store",
            elem_classes="filter",
        )
        theme_toggle = gr.Button("Toggle Light/Dark Theme", elem_classes="theme-toggle")

    # Main tabs
    with gr.Tabs(elem_id="main-tabs") as tabs:
        # Query tab
        with gr.TabItem("Ask Question"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Input box
                    question_input = gr.Textbox(
                        label="Ask a retail-related question",
                        placeholder="e.g., What are the top 5 selling products in March?",
                        lines=3,
                    )

                    # Previous questions suggestions
                    with gr.Accordion("Example Questions", open=False):
                        example_questions = gr.Dataset(
                            components=[gr.Textbox(visible=False)],
                            samples=[
                                ["What are the top 5 selling products in March?"],
                                ["Show me total sales for each category"],
                                ["Which store has the highest revenue?"],
                                ["What is the most popular payment method?"],
                                ["Compare sales between different categories"],
                                ["Show me daily sales trend"],
                                ["Who are our top spending customers?"],
                            ],
                            label="Click to use an example question",
                        )

                    # Buttons
                    with gr.Row():
                        clear_btn = gr.Button("Clear", elem_classes="secondary")
                        submit_btn = gr.Button("Submit", variant="primary")

                with gr.Column(scale=2):
                    # Metrics cards
                    with gr.Row():
                        with gr.Column():
                            total_revenue_metric = gr.Markdown(
                                "### $0", elem_classes="metric-card"
                            )
                            gr.Markdown("Total Revenue", elem_classes="metric-label")

                        with gr.Column():
                            total_transactions_metric = gr.Markdown(
                                "### 0", elem_classes="metric-card"
                            )
                            gr.Markdown(
                                "Total Transactions", elem_classes="metric-label"
                            )

                    with gr.Row():
                        with gr.Column():
                            avg_value_metric = gr.Markdown(
                                "### $0", elem_classes="metric-card"
                            )
                            gr.Markdown(
                                "Avg Transaction Value", elem_classes="metric-label"
                            )

                        with gr.Column():
                            top_category_metric = gr.Markdown(
                                "### N/A", elem_classes="metric-card"
                            )
                            gr.Markdown("Top Category", elem_classes="metric-label")

        # Results tab
        with gr.TabItem("Results"):
            with gr.Row():
                # Left column: Results table and SQL
                with gr.Column(scale=3):
                    results_table = gr.DataFrame(label="Query Result")
                    export_btn = gr.Button(
                        "Export Data (CSV)", elem_classes="download-button"
                    )
                    export_data_link = gr.HTML(
                        '<a href="#" download="data.csv" id="download-link" style="display:none">Download CSV</a>'
                    )
                    predicted_intent = gr.Textbox(label="Predicted Intent")
                    generated_sql = gr.Textbox(label="Generated SQL")

                    # Visualization options
                    with gr.Row(elem_classes="visualization-controls"):
                        gr.Markdown("### Visualization Options")
                        chart_type = gr.Radio(
                            ["auto", "bar", "line", "pie", "horizontal", "scatter"],
                            label="Chart Type",
                            value="auto",
                        )
                        update_chart_btn = gr.Button("Update Visualization")

                # Right column: Chart and insights
                with gr.Column(scale=2):
                    chart_output = gr.Image(label="Chart")

                    # Insights panel
                    gr.Markdown("### Key Insights")
                    insights_panel = gr.HTML(label="")

        # Data Explorer tab (new)
        with gr.TabItem("Data Explorer"):
            with gr.Row():
                data_explorer_table = gr.DataFrame(label="Raw Data")

                with gr.Column():
                    gr.Markdown("### Store Information")
                    store_info = gr.JSON(
                        value={
                            "1": {
                                "name": "Downtown Store",
                                "location": "123 Main St",
                                "manager": "John Smith",
                            },
                            "2": {
                                "name": "Shopping Mall",
                                "location": "456 Market Ave",
                                "manager": "Jane Doe",
                            },
                            "3": {
                                "name": "Outlet Center",
                                "location": "789 Commerce Blvd",
                                "manager": "Robert Johnson",
                            },
                        }
                    )

    # Footer
    gr.Markdown("Built with Gradio • RetailGenie v2.0", elem_classes="footer")

    # Handle example question selection
    example_questions.click(
        lambda x: x[0], inputs=[example_questions], outputs=[question_input]
    )

    # Format insights HTML
    def format_insights_html(insights):
        if not insights:
            return "<p>No insights available for this query.</p>"

        html = "<div class='insights-container'>"
        for insight in insights:
            html += f"<div class='insight-card'>{insight}</div>"
        html += "</div>"
        return html

    # Update CSV download link
    def update_download_link(link):
        if not link:
            return ""
        return f'<a href="{link}" download="retail_data.csv" id="download-link" class="download-link">Click here to download CSV</a>'

    # Update metrics display
    def update_metrics_display(metrics):
        return (
            f"### {metrics['total_revenue']}",
            f"### {metrics['total_transactions']}",
            f"### {metrics['avg_value']}",
            f"### {metrics['top_category']}",
        )

    # Set up interactivity
    # Main query submission
    submit_btn.click(
        fn=retailgenie_pipeline,
        inputs=[question_input, date_filter, store_filter, chart_type, theme_state],
        outputs=[
            results_table,
            export_data_link,
            predicted_intent,
            generated_sql,
            chart_output,
            insights_panel,
            gr.State(),  # for metrics
            data_explorer_table,
        ],
    ).then(
        fn=format_insights_html, inputs=[insights_panel], outputs=[insights_panel]
    ).then(
        fn=update_download_link, inputs=[export_data_link], outputs=[export_data_link]
    ).then(
        fn=update_metrics_display,
        inputs=[gr.State()],
        outputs=[
            total_revenue_metric,
            total_transactions_metric,
            avg_value_metric,
            top_category_metric,
        ],
    ).then(
        fn=switch_to_results_tab, inputs=[], outputs=[tabs]
    )

    # Update chart based on chart type selection
    update_chart_btn.click(
        fn=retailgenie_pipeline,
        inputs=[question_input, date_filter, store_filter, chart_type, theme_state],
        outputs=[
            results_table,
            export_data_link,
            predicted_intent,
            generated_sql,
            chart_output,
            insights_panel,
            gr.State(),  # for metrics
            data_explorer_table,
        ],
    ).then(
        fn=format_insights_html, inputs=[insights_panel], outputs=[insights_panel]
    ).then(
        fn=update_download_link, inputs=[export_data_link], outputs=[export_data_link]
    )

    # Clear button
    clear_btn.click(
        lambda: [
            "",
            None,
            "",
            "",
            None,
            None,
            {
                "total_revenue": "$0",
                "total_transactions": "0",
                "avg_value": "$0",
                "top_category": "N/A",
            },
            None,
        ],
        inputs=[],
        outputs=[
            question_input,
            results_table,
            predicted_intent,
            generated_sql,
            chart_output,
            insights_panel,
            gr.State(),  # for metrics
            data_explorer_table,
        ],
    ).then(
        fn=lambda: ["### $0", "### 0", "### $0", "### N/A"],
        inputs=[],
        outputs=[
            total_revenue_metric,
            total_transactions_metric,
            avg_value_metric,
            top_category_metric,
        ],
    )

    # Theme toggle
    theme_toggle.click(fn=toggle_theme, inputs=[theme_state], outputs=[theme_state])

if __name__ == "__main__":
    demo.launch()
