# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold

from file_interface import load_from_csv, write_submission

from features import make_features, write_features_file
from evaluation import proba_to_player, accuracy, k_fold_cv_score
from tuning import tuning

from models import tree

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
        >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


if __name__ == '__main__':
    
    # ------------------------------- Tuning ------------------------------- #
    
    tuning = True
    if tuning == True:
        print("Tuning in process...")
        
        X_tuning = load_from_csv('data/input_train_set.csv')
        y_tuning = load_from_csv('data/output_train_set.csv')
        
        model, parameter = tuning(X_tuning, y_tuning)     

    
        print(f"The best model is {model} with parameter {parameter}. ")
        
        
    # ------------------------------- Learning ------------------------------- #
    
    learning = False
    if learning == True:
        print("Loading data...")
    
        X_LS = load_from_csv('data/input_train_set.csv')
        y_LS = load_from_csv('data/output_train_set.csv')
        
        nb_passes_LS = X_LS.shape[0]
    
        print("Done.")
        print("Deriving features...")
    
        # X_features : (nb_samples, nb_features)
        new_features = True
        if new_features is True:
            X_LS_pairs, y_LS_pairs = make_features(X_LS, y_LS)
            write_features_file(X_LS_pairs, "X_LS")
            write_features_file(y_LS_pairs, "y_LS")
        else:
            X_LS_pairs = load_from_csv(os.path.join("features", "X_LS.csv"))
            y_LS_pairs = load_from_csv(os.path.join("features", "y_LS.csv"))
    
        print("Done.")
    
        print("Learning with a simple tree...")
        
        print(X_LS_pairs)
        clf = tree(X_LS_pairs, y_LS_pairs)
        
    
        print("Done.")

    
    # ------------------------------- Cross validation ------------------------------- #
    
        print("Evalution of the accuracy with cross validation...")
        
        k = 5     
        max_depth = 10
        cv_score = k_fold_cv_score(X_LS_pairs, y_LS_pairs, k, tree, max_depth)
        
        print(cv_score)
        
        print("Done.")  
    

    # ------------------------------ Prediction ------------------------------ #
    
        # Load test data
        X_TS = load_from_csv('data/input_test_set.csv')
        nb_passes_TS = X_TS.shape[0]
    
        # Same transformation as LS
        new_features = True
        if new_features is True:
            X_TS_pairs, _ = make_features(X_TS)
            write_features_file(X_TS_pairs, "X_TS")
        else:
            X_TS_pairs = load_from_csv(os.path.join("features", "X_TS.csv"))
    
        # Predict
        proba_pairs = clf.predict_proba(X_TS_pairs)
        player_predictions, proba_passes = proba_to_player(proba_pairs)
        
        # Estimated score of the model
        predicted_score = cv_score
    
        # Making the submission file
        fname = write_submission(predictions=player_predictions, probas=proba_passes, estimated_score=predicted_score, file_name="results/submission")
        print('Submission file "{}" successfully written'.format(fname))
