import logging
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Configure logging using logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineering:
    def __init__(self, k_best_features=5, n_components_pca=2):
        """
        Initialize feature engineering parameters.
        :param k_best_features: Number of top features to select using SelectKBest.
        :param n_components_pca: Number of components for PCA dimensionality reduction.
        """
        self.k_best_features = k_best_features
        self.n_components_pca = n_components_pca

    def create_new_features(self, df):
        """
        Generate new features from existing data.
        Example: Create interaction terms or extract date-related features.
        """
        try:
            # Example feature: sum of numerical features (interaction term)
            df['sum_feature'] = df.select_dtypes(include=['float64', 'int64']).sum(axis=1)
            logging.info("New features created")
            return df
        except Exception as e:
            logging.error(f"Error while creating new features: {e}")
            raise

    def encode_categorical_variables(self, df):
        """
        Encode categorical variables using Label Encoding.
        :param df: DataFrame with features.
        :return: DataFrame with encoded categorical features.
        """
        try:
            label_encoder = LabelEncoder()
            cat_cols = df.select_dtypes(include=['object']).columns

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])

            logging.info("Categorical variables encoded")
            return df
        except Exception as e:
            logging.error(f"Error while encoding categorical variables: {e}")
            raise

    def select_best_features(self, X, y):
        """
        Perform feature selection using SelectKBest.
        :param X: Input features.
        :param y: Target variable.
        :return: DataFrame with selected features.
        """
        try:
            # Make sure all features are numeric
            X = self.encode_categorical_variables(X)

            # Apply SelectKBest
            selector = SelectKBest(score_func=chi2, k=self.k_best_features)
            X_new = selector.fit_transform(X, y)
            logging.info(f"Top {self.k_best_features} features selected")
            return pd.DataFrame(X_new)
        except Exception as e:
            logging.error(f"Error during feature selection: {e}")
            raise

    def apply_pca(self, X):
        """
        Apply Principal Component Analysis for dimensionality reduction.
        :param X: Input features.
        :return: DataFrame with transformed features.
        """
        try:
            pca = PCA(n_components=self.n_components_pca)
            X_pca = pca.fit_transform(X)
            logging.info(f"PCA applied with {self.n_components_pca} components")
            return pd.DataFrame(X_pca)
        except Exception as e:
            logging.error(f"Error during PCA: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    file_path = "../data/customer_activity.csv"  # Update if needed
    target_column = "Debt_Status"  # Replace with actual target column

    # Load data
    try:
        df = load_data(file_path)
        logging.info("Data loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        exit(1)

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Initialize FeatureEngineering class
    feature_engineer = FeatureEngineering(k_best_features=5, n_components_pca=2)

    # Create new features
    try:
        df = feature_engineer.create_new_features(df)
    except Exception as e:
        logging.error(f"Failed to create new features: {e}")
        exit(1)

    # Feature selection
    try:
        X_selected = feature_engineer.select_best_features(X, y)
        logging.info(f"Selected features shape: {X_selected.shape}")
    except Exception as e:
        logging.error(f"Failed to select best features: {e}")
        exit(1)

    # Apply PCA
    try:
        X_pca = feature_engineer.apply_pca(X)
        logging.info(f"PCA-transformed features shape: {X_pca.shape}")
    except Exception as e:
        logging.error(f"Failed to apply PCA: {e}")
        exit(1)
