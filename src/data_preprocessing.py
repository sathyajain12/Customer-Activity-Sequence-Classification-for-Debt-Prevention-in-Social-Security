import os
from logger import logger
from data_loader import load_data, display_data_info
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Configure logging


class DataPreprocessing:
    def __init__(self, file_path, target_column, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def preprocess_data(self, df):
        try:
            # Display the data info
            display_data_info(df)

            # Handle missing values, encode categorical variables, and scale numerical features
            df = self.handle_missing_values(df)
            df = self.encode_categorical_variables(df)
            df = self.scale_features(df)

            # Split the data into features and target
            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            logger.info("Data split into training and testing sets")

            return X_train, X_test, y_train, y_test
        except KeyError:
            logger.error(f"Target column '{self.target_column}' not found in the dataset")
            raise
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def handle_missing_values(self, df):
        df = df.dropna(subset=[self.target_column])
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        cat_cols = df.select_dtypes(include=['object']).columns
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

        logger.info("Missing values handled")
        return df

    def encode_categorical_variables(self, df):
        label_encoder = LabelEncoder()
        cat_cols = df.select_dtypes(include=['object']).columns

        for col in cat_cols:
            df[col] = label_encoder.fit_transform(df[col])

        logger.info("Categorical variables encoded")
        return df

    def scale_features(self, df):
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns

        df[num_cols] = scaler.fit_transform(df[num_cols])

        logger.info("Numerical features scaled")
        return df

if __name__ == "__main__":
    # File path and target column definition
    file_path = "C:/Users/sathy/OneDrive/Desktop/Social security/Customer-Activity-Sequence-Classification-for-Debt-Prevention-in-Social-Security/data/customer_activity.csv " 
    target_column = "Debt_Status" 

    # Loading data using data_loader.py
    try:
        df = load_data(file_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        exit(1)

    # Initializing the DataPreprocessing class
    data_preprocessor = DataPreprocessing(file_path, target_column)

    # Preprocessing the data
    try:
        X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data(df)
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Testing features shape: {X_test.shape}")
        logger.info(f"Training target shape: {y_train.shape}")
        logger.info(f"Testing target shape: {y_test.shape}")
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
