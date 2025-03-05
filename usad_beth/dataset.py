from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import sys

import requests
import zipfile
import numpy as np
import pandas as pd

from usad_beth.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


TRAINING_FILE = "labelled_training_data.csv"
VALIDATION_FILE = "labelled_validation_data.csv"
TESTING_FILE = "labelled_testing_data.csv"

def download_dataset(url="https://www.kaggle.com/api/v1/datasets/download/katehighnam/beth-dataset",
                     path=RAW_DATA_DIR / "beth-dataset.zip"):
    r = requests.get(url)
    with open(path, "wb") as f:
        f.write(r.content)

def extract_dataset(path = RAW_DATA_DIR / "beth-dataset.zip"):
    with zipfile.ZipFile(path, "r") as zf:
        zf.extract(TRAINING_FILE, RAW_DATA_DIR)
        zf.extract(TESTING_FILE, RAW_DATA_DIR)
        zf.extract(VALIDATION_FILE, RAW_DATA_DIR)

@app.command()
def main():        
    logger.info("Downloading dataset...")
    download_dataset()
    logger.success("Downloading dataset completed.")
    
    logger.info("Extracting dataset...")
    extract_dataset()
    logger.success("Extracting dataset completed.")


if __name__ == "__main__":
    app()
