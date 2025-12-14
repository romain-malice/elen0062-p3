# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from file_interface import load_from_csv, write_submission

from models import test_some_ks_on_knn, test_trees, test_svm
from features import make_features

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

    print("Done.")
    print("Deriving features...")

    # X_features : (nb_samples, nb_features)
    X_features, y_features = make_features(X_LS, y_LS)
    y_features = y_features.to_numpy(dtype=np.float64).ravel()

    print("Done.")

    print("Learning with a simple tree...")

    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_features, y_features)

    print("Done.")

    
    # ------------------------------- Cross validation ------------------------------- #

    knn = False
    if knn is True:
        print("\nTesting knn...")
        k_values = np.array([50, 100, 200, 500])
        print(f"Tested values of k: {k_values}")
        means, variances = test_some_ks_on_knn(k_values, X_features, y_features, 5)
        print(f"Mean accuracies:\n{means}")
        print(f"Accuracies variance:\n{variances}")
        print("Done.")

    tree = False
    if tree is True:
        print("\nTesting trees...")
        depths = np.array([5, 10, 15, 20])
        print(f"Tested tree depths: {depths}")
        means, variances = test_trees(depths, X_features, y_features, 5)
        print(f"Mean accuracies:\n{means}")
        print(f"Accuracies variance:\n{variances}")
        print("Done.")

    svm = False
    if svm is True:
        print("Testing SVM...")
        means, variances = test_svm(X_features, y_features, 5)
        print(f"Mean accuracies:\n{means}")
        print(f"Accuracies variance:\n{variances}")
        print("Done.")

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    X_TS = load_from_csv('data/input_test_set.csv')

    # Same transformation as LS
    X_TS_pairs, _ = make_features(X_TS)
    print(X_TS_pairs)

    #X_TS_pairs["distance"] = compute_distance_(X_TS_pairs)

    #X_TS_features = X_TS_pairs[["distance", "same_team"]]


    # Predict
    #y_pred = model.predict_proba(X_TS_features)[:,1]

    # Deriving probas
    #probas = y_pred.reshape(X_TS.shape[0], 22)

    # Estimated score of the model
    #predicted_score = 0.01 # it is quite logical...

    # Making the submission file
    #fname = write_submission(probas=probas, estimated_score=predicted_score, file_name="results/toy_example_probas")
    #print('Submission file "{}" successfully written'.format(fname))

    # -------------------------- Random Prediction -------------------------- #

    #random_state = 0
    #random_state = check_random_state(random_state)
    #predictions = random_state.choice(np.arange(1,23), size=X_TS.shape[0], replace=True)

    #fname = write_submission(predictions=predictions, estimated_score=predicted_score, file_name="results/toy_example_predictions")
    #print('Submission file "{}" successfully written'.format(fname))
