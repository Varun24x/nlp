import json
import pandas as pd
import os


# Function to load and preprocess datasets
def load_data(train_path, dev_path, custom_path=None):
    """
    Loads the Spider dataset and optional custom dataset from provided JSON file paths.
    
    :param train_path: Path to the Spider training data (JSON format).
    :param dev_path: Path to the Spider development data (JSON format).
    :param custom_path: (Optional) Path to the custom data (JSON format).
    :return: A tuple containing the NL queries and SQL queries for train, dev, and custom sets.
    """
    def load_json_file(path):
        """Load a JSON file and return the data."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found - {path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file - {path}")
            return []

    # Load datasets
    train_data = load_json_file(train_path)
    dev_data = load_json_file(dev_path)
    custom_data = load_json_file(custom_path) if custom_path else []

    # Extract NL-SQL pairs from the data
    def extract_data(data):
        if not data:
            return [], []
        nl_queries, sql_queries = [], []
        for item in data:
            nl_queries.append(item.get("question", ""))
            sql_queries.append(item.get("query", ""))
        return nl_queries, sql_queries

    # Extract queries
    train_nl, train_sql = extract_data(train_data)
    dev_nl, dev_sql = extract_data(dev_data)
    custom_nl, custom_sql = extract_data(custom_data)

    return train_nl, train_sql, dev_nl, dev_sql, custom_nl, custom_sql


# Paths to the dataset files
train_path = os.path.join("data", "train_spider.json")  # Spider training JSON data
dev_path = os.path.join("data", "dev.json")             # Spider development JSON data
custom_path = os.path.join("data", "custom.json")       # Custom JSON data

# Load and preprocess data
train_nl, train_sql, dev_nl, dev_sql, custom_nl, custom_sql = load_data(train_path, dev_path, custom_path)

# Create DataFrames for training, development, and custom data
train_df = pd.DataFrame({"nl_query": train_nl, "sql_query": train_sql})
dev_df = pd.DataFrame({"nl_query": dev_nl, "sql_query": dev_sql})

if custom_nl and custom_sql:  # Only create DataFrame if custom data is available
    custom_df = pd.DataFrame({"nl_query": custom_nl, "sql_query": custom_sql})
    print(f"Custom dataset loaded: {len(custom_nl)} entries.")
else:
    custom_df = pd.DataFrame(columns=["nl_query", "sql_query"])
    print("No custom data found or custom dataset is empty.")

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Save the data into CSV files for easy loading later during training
train_df.to_csv(os.path.join("data", "train_data.csv"), index=False)
dev_df.to_csv(os.path.join("data", "dev_data.csv"), index=False)
custom_df.to_csv(os.path.join("data", "custom_data.csv"), index=False)

print("Preprocessing complete. Data saved to CSV.")
