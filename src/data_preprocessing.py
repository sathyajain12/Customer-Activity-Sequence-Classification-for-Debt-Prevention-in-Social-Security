import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessing:
    def __init__(self, file_path, target_column, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully from {self.file_path}")
            return df
        except FileNotFoundError:
            logging.error(f"File not found: {self.file_path}")
            raise
        except pd.errors.EmptyDataError:
            logging.error(f"No data found in the file: {self.file_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, df):
        try:
            df = self.handle_missing_values(df)
            df = self.encode_categorical_variables(df)
            df = self.scale_features(df)

            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            logging.info("Data split into training and testing sets")

            return X_train, X_test, y_train, y_test
        except KeyError:
            logging.error(f"Target column '{self.target_column}' not found in the dataset")
            raise
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def handle_missing_values(self, df):
        df = df.dropna(subset=[self.target_column])
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        cat_cols = df.select_dtypes(include=['object']).columns
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

        logging.info("Missing values handled")
        return df

    def encode_categorical_variables(self, df):
        label_encoder = LabelEncoder()
        cat_cols = df.select_dtypes(include=['object']).columns

        for col in cat_cols:
            df[col] = label_encoder.fit_transform(df[col])

        logging.info("Categorical variables encoded")
        return df

    def scale_features(self, df):
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns

        df[num_cols] = scaler.fit_transform(df[num_cols])

        logging.info("Numerical features scaled")
        return df

if __name__ == "__main__":
    file_path = "C:/Users/sathy/OneDrive/Desktop/Social security/Customer-Activity-Sequence-Classification-for-Debt-Prevention-in-Social-Security/data/customer_activity.csv"  # Make sure this path is correct
    target_column = "Debt_Status"  # Replace with the actual name of the target column

    # Initialize the DataPreprocessing class
    data_preprocessor = DataPreprocessing(file_path, target_column)

    # Load the data
    try:
        df = data_preprocessor.load_data()
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        exit(1)

    # Preprocess the data
    try:
        X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data(df)
        logging.info(f"Training features shape: {X_train.shape}")
        logging.info(f"Testing features shape: {X_test.shape}")
        logging.info(f"Training target shape: {y_train.shape}")
        logging.info(f"Testing target shape: {y_test.shape}")
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")
