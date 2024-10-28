import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}. Error: {e}")
        raise e

def display_data_info(df):
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Dataset columns: {df.columns.tolist()}")
    logging.info(f"Missing values in dataset:\n{df.isnull().sum()}")
    logging.info(f"Data preview:\n{df.head()}")

# Example usage
if __name__ == "__main__":
    file_path = "C:/Users/sathy/OneDrive/Desktop/Social security/Customer-Activity-Sequence-Classification-for-Debt-Prevention-in-Social-Security/data/customer_activity.csv"  # Update this path if needed
    data = load_data(file_path)
    display_data_info(data)
