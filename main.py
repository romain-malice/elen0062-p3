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
        X_LS_features, y_LS_features = make_features(X_LS, y_LS)
        write_features_file(X_LS_features, "X_LS")
        write_features_file(y_LS_features, "y_LS")
        y_LS_features = y_LS_features.to_numpy().ravel()
    else:
        X_LS_features = load_from_csv(os.path.join("features", "X_LS.csv"))
        y_LS_features = load_from_csv(os.path.join("features", "y_LS.csv"))
        y_LS_features = y_LS_features.to_numpy().ravel()

    print("Done.")

    print("Learning with a simple tree...")

    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_LS_features, y_LS_features)
    

    print("Done.")

    
    # ------------------------------- Cross validation ------------------------------- #
    
    print("Evalution of the accuracy with cross validation...")
    
    k = 5
    cv_score = k_fold_cv_score(X_LS_features, y_LS_features, k, )
    
    print("Done.")  
    

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    X_TS = load_from_csv('data/input_test_set.csv')
    nb_passes_TS = X_TS.shape[0]

    # Same transformation as LS
    new_features = True
    if new_features is True:
        X_TS_features, _ = make_features(X_TS)
        write_features_file(X_TS_features, "X_TS")
    else:
        X_TS_features = load_from_csv(os.path.join("features", "X_TS.csv"))

    # Predict
    proba = clf.predict_proba(X_TS_features)
    player = proba_to_player(proba, nb_passes_TS)
    proba = proba[:, 1].reshape(nb_passes_TS, 22) 
    

    # Estimated score of the model
    predicted_score = cv_score

    # Making the submission file
    fname = write_submission(predictions=player, probas=proba, estimated_score=predicted_score, file_name="results/submission_de_francois_et_romain")
    print('Submission file "{}" successfully written'.format(fname))
