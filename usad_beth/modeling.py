from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import time
from collections import namedtuple
import json
import copy
import datetime

import numpy as np
import pandas as pd

from streamad.model import xStreamDetector, RrcfDetector, LodaDetector, HSTreeDetector, RShashDetector
from streamad.process import ZScoreCalibrator

from river import anomaly
from river import stream
from river import compose
from river import preprocessing

from pysad.models import xStream, IForestASD, ExactStorm, LODA, HalfSpaceTrees, KitNet, KNNCAD, RSHash, RobustRandomCutForest, LocalOutlierProbability, MedianAbsoluteDeviation

from sklearn.metrics import roc_auc_score

from usad_beth.config import PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

THRESHOLD = 0.95
SEED = 2025
NUM_SEEDS=5

# List of algorithms for experimenting
MODELNAMES = ["OCSVM",
               "HSTree", "ILOF",
              "RShash", "xStream", "RRCF", "LODA",
              "IForestASD", "KitNet", "Storm"]

# Function for evaluating model results
def fit_score_model(model,
               X_train, X_test,
               scaler = None,
               encoder = None,
               probability_threshold = THRESHOLD):
    calibrator = ZScoreCalibrator(sigma=3, extreme_sigma=5, is_global=True)
    
    preds = []
    scores = []
    normalized_scores = []
    duration = []
    
    module = model.__module__    

    start = time.time()
    
    for x, _  in tqdm(stream.iter_pandas(X_train)):     
        score = None

        if encoder != None and 'processName' in x and 'parentProcessName' in x:
            encoder.learn_one(x)            
            x = encoder.transform_one(x)

        if scaler != None:
            scaler.learn_one(x)
            x = scaler.transform_one(x)

        if module.startswith("river"):        
            model.learn_one(x)        
            score = model.score_one(x) 
        elif module.startswith("pysad"):        
            X = np.array(list(x.values()))            
            model.fit_partial(X)  
            score = model.score_partial(X)            
        elif module.startswith("streamad"):        
            score = model.fit_score(np.array(list(x.values())))

        if type(score) == np.ndarray:
            score = score[0]                
        calibrator.normalize(score)
        
    for x, _ in tqdm(stream.iter_pandas(X_test)):
        score = None
        is_anomaly = False

        if encoder != None and 'processName' in x and 'parentProcessName' in x:
            encoder.learn_one(x)            
            x = encoder.transform_one(x)

        if scaler != None:
            scaler.learn_one(x)
            x = scaler.transform_one(x)

        if module.startswith("river"):        
            score = model.score_one(x)               
            is_anomaly = model['filter'].classify(score)
            model.learn_one(x)
        elif module.startswith("pysad"):        
            X = np.array(list(x.values()))                        
            score = model.fit_score_partial(X)            
        elif module.startswith("streamad"):        
            score = model.fit_score(np.array(list(x.values())))            

        if type(score) == np.ndarray:
            score = score[0]
                                
        normalized_score = calibrator.normalize(score)
        if normalized_score!= None and normalized_score > probability_threshold and \
            not module.startswith("river"):
            is_anomaly = True
        
        duration = time.time() - start

        scores.append(score)
        normalized_scores.append(normalized_score)
        preds.append(is_anomaly)
                
    return {'preds': preds,
            'scores': scores,
            'normalized_scores': normalized_scores,
            'time': duration}

# Function for calculating rocauc score
def get_roc_auc(y_true: list, scores: list, invert = False) -> float:
    result = roc_auc_score(np.array(y_true, dtype=bool), np.array(scores))
    is_inverted = False    
    if result < 0.5:
        result_inverted = roc_auc_score(np.array(y_true, dtype=bool), -np.array(scores)) 
        print("Reverted: ", result_inverted, result+result_inverted)
        if invert:
            is_inverted = True
            result = result_inverted                
    return result, is_inverted

# Function for writing latest evaluation results to file
def write_results(eval_results: namedtuple):    
    f = open(REPORTS_DIR / "eval_results.txt", "a")    
    res = eval_results[-1]
    f.write(json.dumps(res._asdict()) + "\n")                    
    f.close()



####################################################################################
# Class for initializing new model instances, without knowledge of previous fittings
####################################################################################
class ModelStock:
    def __init__(self):        
        self.model_params = {
            "OCSVM" : {},
            "HSTree" : {},
            "ILOF" : {},
            "RShash" : {}, 
            "xStream" : {}, 
            "IForestASD" : {},
            "RRCF" : {},            
            "KitNet" : {},
            "Storm" : {},
            "LODA" : {}
        }

        self.models =  { 
                "OCSVM": compose.Pipeline(
                    ('scale', preprocessing.StandardScaler()),
                    ('filter', anomaly.ThresholdFilter(
                        anomaly.OneClassSVM(),
                        threshold = THRESHOLD 
                    ))),
                "HSTree": compose.Pipeline(
                    ('scale', preprocessing.MinMaxScaler()),                    
                    ('filter', anomaly.ThresholdFilter(
                        anomaly.HalfSpaceTrees(),
                        threshold = THRESHOLD 
                    ))),
                "ILOF": compose.Pipeline(                    
                        ('filter', anomaly.ThresholdFilter(
                            anomaly.LocalOutlierFactor(),
                            threshold = THRESHOLD 
                    ))),
                
                "RShash": RShashDetector(**self.model_params["RShash"]),
                "xStream": xStreamDetector(**self.model_params["xStream"]),             
                "RRCF": RrcfDetector(**self.model_params["RRCF"]),
                "LODA": LodaDetector(**self.model_params["LODA"]),

                # "RRCF": RobustRandomCutForest(**models_params["RRCF"]),
                # "xStream": xStream(**models_params["xStream"]), 
                # "LODA": LODA(**models_params["LODA"])

                "IForestASD": IForestASD(**self.model_params["IForestASD"]),                 
                "KitNet": KitNet(**self.model_params["KitNet"]),
                "Storm": ExactStorm(**self.model_params["Storm"]),                
        }

    def model(self, model_name: str):
        try:    
            return copy.deepcopy(self.models[model_name])
        except:
            if str(self.models[model_name].__class__).find("RrcfDetector") != -1:
                return RrcfDetector(**self.model_params["RRCF"])
            else:
                return None
####################################################################################


def run_experiment(modelname: str,
                   X_train: pd.DataFrame, X_test: pd.DataFrame, encoder=None,
                   seed=0):
    model = ModelStock().model(modelname)
    np.random.seed(seed)
    results = fit_score_model(model, X_train, X_test, encoder=encoder)
    return results
    

@app.command()
def main():
    Eval_result = namedtuple("Evaluation_result", "Algorithm Seed Sus_or_evil Sorted_train Enriched_parent_process_name Time ROCAUC_score")
    eval_results = []
    
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv", index_col=0)
    X_train_sorted = pd.read_csv(PROCESSED_DATA_DIR / "X_train_sorted.csv", index_col=0)
    X_train_enriched = pd.read_csv(PROCESSED_DATA_DIR / "X_train_enriched.csv", index_col=0)
    X_train_sorted_enriched = pd.read_csv(PROCESSED_DATA_DIR / "X_train_sorted_enriched.csv", index_col=0)

    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv", index_col=0)
    X_test_enriched = pd.read_csv(PROCESSED_DATA_DIR / "X_test_enriched.csv", index_col=0)

    y_test_sus = pd.read_csv(PROCESSED_DATA_DIR / "y_test_sus.csv", index_col=0)
    y_test_evil = pd.read_csv(PROCESSED_DATA_DIR / "y_test_evil.csv", index_col=0)

    for s in range(SEED, SEED + NUM_SEEDS):
        logger.info(f'SEED = {s}')
        for name in MODELNAMES:
            logger.info(f'Model {name} evaluation')

            logger.info('Train without sorting and enriching')
            results = run_experiment(name, X_train, X_test, seed=s)
            eval_results.append(Eval_result(name, s, "sus", "no", "no", results["time"], get_roc_auc(y_test_sus, results["scores"])))
            write_results(eval_results)
            eval_results.append(Eval_result(name, s, "evil", "no", "no", results["time"], get_roc_auc(y_test_evil, results["scores"])))
            write_results(eval_results)
            logger.success('Completed')

            logger.info('Train with sorting')
            results = run_experiment(name, X_train_sorted, X_test, seed=s)
            eval_results.append(Eval_result(name, s, "sus", "yes", "no", results["time"], get_roc_auc(y_test_sus, results["scores"])))
            write_results(eval_results)
            eval_results.append(Eval_result(name, s, "evil", "yes", "no", results["time"], get_roc_auc(y_test_evil, results["scores"])))
            write_results(eval_results)
            logger.success('Completed')

            logger.info('Train with enriching')
            results = run_experiment(name, X_train_enriched, X_test_enriched, 
                              encoder=preprocessing.OrdinalEncoder(),
                              seed=s)
            eval_results.append(Eval_result(name, s, "sus", "no", "yes", results["time"], get_roc_auc(y_test_sus, results["scores"])))
            write_results(eval_results)
            eval_results.append(Eval_result(name, s, "evil", "no", "yes", results["time"], get_roc_auc(y_test_evil, results["scores"])))
            write_results(eval_results)
            logger.success('Completed')

            logger.info('Train with sorting and enriching')
            results = run_experiment(name, X_train_sorted_enriched, X_test_enriched, 
                              encoder=preprocessing.OrdinalEncoder(),
                              seed = s)
            eval_results.append(Eval_result(name, s, "sus", "yes", "yes", results["time"], get_roc_auc(y_test_sus, results["scores"])))
            write_results(eval_results)
            eval_results.append(Eval_result(name, s, "evil", "yes", "yes", results["time"], get_roc_auc(y_test_evil, results["scores"])))
            write_results(eval_results)
            logger.success('Completed')
            
            logger.success(f'Model {name} evaluation completed')    
    
    df_eval_results = pd.DataFrame(eval_results)
    now = str(datetime.datetime.now())
    df_eval_results.to_csv(REPORTS_DIR / f"eval_results_{now}.csv")
    
    return df_eval_results

if __name__ == "__main__":
    app()
