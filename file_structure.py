import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of files to create for the simplified project structure
list_of_files = [
    "data/.gitkeep",
    "src/__init__.py",
    "src/data_preprocessing.py",
    "src/feature_engineering.py",
    "src/model_training.py",
    "src/model_evaluation.py",
    "src/prediction.py",
    "tests/__init__.py",
    "experiment/experiments.ipynb",
    "init_setup.sh",
    "requirements.txt",
    "README.md"
]

# Create directories and files based on the list
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    # Create directory if it doesn't exist
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")
    
    # Create the file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass  # Create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
