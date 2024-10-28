from src.data_loader import DataLoader
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation
from src.prediction import Prediction
from logger import logger

def main():
    # Step 1: Load Data
    try:
        file_path = "data/customer_activity.csv"
        data_loader = DataLoader(file_path)
        df = data_loader.load_data()
        logger.info("Data loading completed.")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # Step 2: Data Preprocessing
    try:
        target_column = "Debt_Status"  # Replace with the actual target column name
        data_preprocessor = DataPreprocessing(target_column=target_column)
        X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data(df)
        logger.info("Data preprocessing completed.")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return

    # Step 3: Feature Engineering
    try:
        feature_engineer = FeatureEngineering()
        X_train = feature_engineer.transform(X_train)
        X_test = feature_engineer.transform(X_test)
        logger.info("Feature engineering completed.")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return

    # Step 4: Model Training
    try:
        model_trainer = ModelTraining()
        model = model_trainer.train_model(X_train, y_train)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return

    # Step 5: Model Evaluation
    try:
        model_evaluator = ModelEvaluation()
        model_evaluator.evaluate_model(model, X_test, y_test)
        logger.info("Model evaluation completed.")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return

    # Step 6: Predictions
    try:
        predictor = Prediction()
        predictions = predictor.predict(model, X_test)
        logger.info("Prediction completed.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
