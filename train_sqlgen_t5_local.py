
import argparse
import os
import pandas as pd
from azureml.core import Run

def main():
    # Get the experiment run context
    run = Run.get_context()
    
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Input data')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    # Log the start of the script
    print(f"Starting training script with input: {args.input_data}, output: {args.output_dir}")
    run.log("training_start", "started")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Simulate loading data
    print("Loading data...")
    try:
        # Try to load data from the input dataset
        dataset = run.input_datasets['sqlgen_data']
        df = dataset.to_pandas_dataframe()
        print(f"Loaded {len(df)} records from dataset")
    except Exception as e:
        print(f"Could not load dataset: {str(e)}")
        print("Creating dummy data instead")
        # Create dummy data
        df = pd.DataFrame({
            'text': ['Sample query 1', 'Sample query 2', 'Sample query 3'],
            'label': ['SELECT * FROM products', 'SELECT * FROM customers', 'SELECT * FROM orders']
        })
    
    # Simulate training
    print("Training model...")
    run.log("training_progress", 25)
    
    # Log metrics
    run.log("accuracy", 0.85)
    run.log("f1_score", 0.82)
    
    run.log("training_progress", 50)
    
    # Simulate saving model
    model_path = os.path.join(args.output_dir, "model.txt")
    with open(model_path, "w") as f:
        f.write("This is a dummy model file")
    
    print(f"Model saved to {model_path}")
    run.log("training_progress", 100)
    
    # Finish
    print("Training completed successfully")
    run.log("training_status", "completed")
    
if __name__ == "__main__":
    main()
