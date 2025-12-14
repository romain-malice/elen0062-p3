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

from models import test_some_ks_on_knn, test_trees, test_svm
from features import make_features, write_features_file
from evaluation import proba_to_player, custom_score

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
    
    nb_passes = X_LS.shape[0]

    kf = KFold(n_splits=5)

    print("Done.")
    print("Deriving features...")

    # X_features : (nb_samples, nb_features)
    new_features = False
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
    
    for train, test in kf.split(X_LS, y_LS):
        pass

    print("Done.")

    
    # ------------------------------- Cross validation ------------------------------- #

    knn = False
    if knn is True:
        print("\nTesting knn...")
        k_values = np.array([50, 100, 200, 500])
        print(f"Tested values of k: {k_values}")
        means, variances = test_some_ks_on_knn(k_values, X_features, y_LS_features, 5)
        print(f"Mean accuracies:\n{means}")
        print(f"Accuracies variance:\n{variances}")
        print("Done.")

    tree = False
    if tree is True:
        print("\nTesting trees...")
        depths = np.array([5, 10, 15, 20])
        print(f"Tested tree depths: {depths}")
        means, variances = test_trees(depths, X_features, y_LS_features, 5)
        print(f"Mean accuracies:\n{means}")
        print(f"Accuracies variance:\n{variances}")
        print("Done.")

    svm = False
    if svm is True:
        print("Testing SVM...")
        means, variances = test_svm(X_features, y_LS_features, 5)
        print(f"Mean accuracies:\n{means}")
        print(f"Accuracies variance:\n{variances}")
        print("Done.")

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    X_TS = load_from_csv('data/input_test_set.csv')
    nb_passes_TS = X_TS.shape[0]

    # Same transformation as LS
    new_features = False
    if new_features is True:
        X_TS_features, _ = make_features(X_TS)
        write_features_file(X_TS_features, "X_TS")
    else:
        X_TS_features = load_from_csv(os.path.join("features", "X_TS.csv"))

    # Predict
    proba = clf.predict_proba(X_TS_features)
    player = proba_to_player(proba, nb_passes_TS)
    proba = proba[:, 1].reshape(nb_passes_TS, 22) 
    print(proba)

    # Estimated score of the model
    predicted_score = 0.31

    # Making the submission file
    fname = write_submission(predictions=player, probas=proba, estimated_score=predicted_score, file_name="results/submission_de_francois_et_romain")
    print('Submission file "{}" successfully written'.format(fname))

    # -------------------------- Random Prediction -------------------------- #

    #random_state = 0
    #random_state = check_random_state(random_state)
    #predictions = random_state.choice(np.arange(1,23), size=X_TS.shape[0], replace=True)

    #fname = write_submission(predictions=predictions, estimated_score=predicted_score, file_name="results/toy_example_predictions")
    #print('Submission file "{}" successfully written'.format(fname))
