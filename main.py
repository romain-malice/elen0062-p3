# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from file_interface import load_from_csv, write_submission

from features import make_features, write_features_file
from evaluation import proba_to_player, accuracy, k_fold_cv_score

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
    
def basic_tree(X_LS_pairs, y_LS_pairs):
    max_depth = 10
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_LS_pairs, y_LS_pairs)
    return clf

if __name__ == '__main__':

    # ------------------------------- Learning ------------------------------- #
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
    
    clf = basic_tree(X_LS_pairs, y_LS_pairs)
    

    print("Done.")

    
    # ------------------------------- Cross validation ------------------------------- #
    
    print("Evalution of the accuracy with cross validation...")
    
    k = 5
    cv_score = k_fold_cv_score(X_LS_pairs, y_LS_pairs, k, basic_tree)
    
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
