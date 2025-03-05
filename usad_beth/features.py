from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from usad_beth.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from usad_beth.dataset import TRAINING_FILE, TESTING_FILE

import pandas as pd
import os

def prepare_dataset(df_original: pd.DataFrame, only_evil=False, enrich = False):
    df = df_original.copy()    
    # Map process ID to OS process(0) vs user process(1).
    df['processId_nonOS'] = df['processId'].map(lambda x: 0 if x in [0, 1, 2] else 1)
    # Map parent process ID to OS process(0) vs user process(1).
    df['parentProcessId_nonOS'] = df['parentProcessId'].map(lambda x: 0 if x in [0, 1, 2] else 1)
    # Map user ID to OS(0) vs user(1).
    df['userId_nonOS'] = df['userId'].map(lambda x: 0 if x < 1000 else 1)
    # Map mount access to folder mnt/ (all non-OS users)(0) vs elsewhere(1).
    df['mountNamespace'] = df['mountNamespace'].map(lambda x: 0 if x == 4026531840 else 1)    
    # Map return value to success(0) vs success with returned value(1) vs success with error(2).
    df['returnValue_error'] = df['returnValue'].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))
  
    fields = ['processId', 'parentProcessId', 'userId', 'mountNamespace', 'eventId', 'argsNum', 'returnValue',
            'processId_nonOS', 'parentProcessId_nonOS', 'userId_nonOS', 'returnValue_error']
    if enrich:
        fields.append('processName')
        fields.append('parentProcessName')
    
    X = df[fields].copy()

    y = df['sus'].copy()            
    if only_evil:
        y = df['evil'].copy()
    y[y == 1] = True 
    y[y == 0] = False  
  
    return X, y     

def enrich_parent_process_name(row: pd.DataFrame, process_vocab: dict):
    process_vocab[(row['hostName'], row['processId'])] = row['processName']     
    if (row['hostName'], row['parentProcessId']) in process_vocab.keys():
        row['parentProcessName'] = process_vocab[(row['hostName'], row['parentProcessId'])]
    else:
        row['parentProcessName'] = None
    return row

app = typer.Typer()



@app.command()
def main():
    
    logger.info("Generating features from dataset...")

    train_df  = pd.read_csv(RAW_DATA_DIR / TRAINING_FILE)    
    test_df = pd.read_csv(RAW_DATA_DIR / TESTING_FILE)
    train_df_sorted = train_df.sort_values(['hostName', 'timestamp'], ignore_index=True)

    X_train, y_train_sus = prepare_dataset(train_df)
    _, y_train_evil = prepare_dataset(train_df, only_evil=True)    
    X_test, y_test_sus = prepare_dataset(test_df)
    _, y_test_evil = prepare_dataset(test_df, only_evil=True)

    X_train_sorted, y_train_sus_sorted = prepare_dataset(train_df_sorted)
    _, y_train_evil_sorted = prepare_dataset(train_df_sorted, only_evil=True)

    X_train, y_train_sus = prepare_dataset(train_df)
    _, y_train_evil = prepare_dataset(train_df, only_evil=True)    
    X_test, y_test_sus = prepare_dataset(test_df)
    _, y_test_evil = prepare_dataset(test_df, only_evil=True)

    X_train_sorted, y_train_sus_sorted = prepare_dataset(train_df_sorted)
    _, y_train_evil_sorted = prepare_dataset(train_df_sorted, only_evil=True)
    
    logger.success("Features generation complete.")

    logger.info("Saving features dataframes...")
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv")    
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv")

    y_train_sus.to_csv(PROCESSED_DATA_DIR / "y_train_sus.csv")
    y_train_evil.to_csv(PROCESSED_DATA_DIR / "y_train_evil.csv")    
    y_test_sus.to_csv(PROCESSED_DATA_DIR / "y_test_sus.csv")
    y_test_evil.to_csv(PROCESSED_DATA_DIR / "y_test_evil.csv")

    X_train_sorted.to_csv(PROCESSED_DATA_DIR / "X_train_sorted.csv")
    y_train_sus_sorted.to_csv(PROCESSED_DATA_DIR / "y_train_sus_sorted.csv")
    y_train_evil_sorted.to_csv(PROCESSED_DATA_DIR / "y_train_evil_sorted.csv")

    logger.success("Saving features dataframes completed.")

    logger.info("Enriching train dataframe with parent process name...")
    process_vocab = {}
    train_df_enriched = train_df.apply(enrich_parent_process_name, process_vocab=process_vocab, axis=1)
    logger.success("Enriching completed.")

    logger.info("Enriching test dataframe with parent process name...")
    test_df_enriched = test_df.apply(enrich_parent_process_name, process_vocab=process_vocab, axis=1)
    logger.success("Enriching completed.")

    logger.info("Enriching sorted train dataframe with parent process name...")
    process_vocab = {}
    train_df_sorted_enriched = train_df_sorted.apply(enrich_parent_process_name, process_vocab=process_vocab, axis=1)
    logger.success("Enriching completed.")

    logger.info("Saving enriched features dataframes...")
    X_train_enriched = None
    X_val_enriched = None
    X_test_enriched = None
    X_train_sorted_enriched = None

    if os.path.exists(PROCESSED_DATA_DIR / "X_train_enriched.csv"):
        X_train_enriched = pd.read_csv(PROCESSED_DATA_DIR / "X_train_enriched.csv", index_col=0)
    else:
        X_train_enriched, _ = prepare_dataset(train_df_enriched, enrich=True)    

    if os.path.exists(PROCESSED_DATA_DIR / "X_test_enriched.csv"):
        X_test_enriched = pd.read_csv(PROCESSED_DATA_DIR / "X_test_enriched.csv", index_col=0)
    else:
        X_test_enriched, _ = prepare_dataset(test_df_enriched, enrich=True)

    if os.path.exists(PROCESSED_DATA_DIR / "X_train_sorted_enriched.csv"):
        X_train_sorted_enriched = pd.read_csv(PROCESSED_DATA_DIR / "X_train_sorted_enriched.csv", index_col=0)
    else:
        X_train_sorted_enriched, _= prepare_dataset(train_df_sorted_enriched, enrich=True)

    X_train_enriched.to_csv(PROCESSED_DATA_DIR / "X_train_enriched.csv")        
    X_test_enriched.to_csv(PROCESSED_DATA_DIR / "X_test_enriched.csv")
    X_train_sorted_enriched.to_csv(PROCESSED_DATA_DIR / "X_train_sorted_enriched.csv")

    logger.success("Saving enriched features dataframes completed.")
        
if __name__ == "__main__":
    app()
