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
from features import make_pair_of_players, compute_distance_

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
    # Load training data
    X_LS = load_from_csv('data/input_train_set.csv')
    y_LS = load_from_csv('data/output_train_set.csv')
    print(X_LS)

    # Transform data as pair of players
    # !! This step is only one way of addressing the problem.
    # We strongly recommend to also consider other approaches that the one provided here.

    X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)

    X_features = X_LS_pairs[["distance", "same_team"]]

    # Build the model
    model = DecisionTreeClassifier()

    with measure_time('Training'):
        print('Training...')
        model.fit(X_features, y_LS_pairs)

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    X_TS = load_from_csv('data/input_test_set.csv')

    # Same transformation as LS
    X_TS_pairs, _ = make_pair_of_players(X_TS)
    X_TS_pairs["distance"] = compute_distance_(X_TS_pairs)

    X_TS_features = X_TS_pairs[["distance", "same_team"]]

    # Predict
    y_pred = model.predict_proba(X_TS_features)[:,1]

    # Deriving probas
    probas = y_pred.reshape(X_TS.shape[0], 22)

    # Estimated score of the model
    predicted_score = 0.01 # it is quite logical...

    # Making the submission file
    fname = write_submission(probas=probas, estimated_score=predicted_score, file_name="results/toy_example_probas")
    print('Submission file "{}" successfully written'.format(fname))

    # -------------------------- Random Prediction -------------------------- #

    random_state = 0
    random_state = check_random_state(random_state)
    predictions = random_state.choice(np.arange(1,23), size=X_TS.shape[0], replace=True)

    fname = write_submission(predictions=predictions, estimated_score=predicted_score, file_name="results/toy_example_predictions")
    print('Submission file "{}" successfully written'.format(fname))
