# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, KFold

from file_interface import load_from_csv, write_submission

from features import make_features, write_features_file
from evaluation import proba_to_player, accuracy, k_fold_cv_score, split_dataset
from tuning import tuning

from models import tree, knn, random_forest, gradient_boosting

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*swapaxes.*",
    category=FutureWarning
)

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
        
    # ------------------------------- Features extraction ------------------------------- #
    print("Getting features...")

    X_LS = load_from_csv('data/input_train_set.csv')
    y_LS = load_from_csv('data/output_train_set.csv')
    
    nb_passes_LS = X_LS.shape[0]

    # X_features : (nb_samples, nb_features)
    new_features = False
    if new_features is True:
        print("Computing features...")
        with measure_time('Features'):
            X_LS_pairs, y_LS_pairs = make_features(X_LS, y_LS)
            write_features_file(X_LS_pairs, "X_LS")
            write_features_file(y_LS_pairs, "y_LS")
    else:
        print("Loading features from file...")
        X_LS_pairs = load_from_csv(os.path.join("features", "X_LS.csv"))
        y_LS_pairs = load_from_csv(os.path.join("features", "y_LS.csv"))

    print("Done.")
    # ------------------------------- Features selection ------------------------------- #
    
    feature_selection = True
    if feature_selection is True:
        print("Feature selection in progress...")

        n_estimators = 105
        max_depth = 8
        k = 5
        clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
        parts_X, parts_y = split_dataset(X_LS_pairs, y_LS_pairs, k)
        with measure_time("Feature selection"):
            importances = np.zeros([k, len(X_LS_pairs.columns)])
            for i, (X, y) in enumerate(zip(parts_X, parts_y)):
                print(f"Starting evaluation on fold {i + 1} out of {k}")
                clf.fit(X_LS_pairs, y_LS_pairs.values.ravel())
                importances[i] = clf.feature_importances_
        importances = np.mean(importances, axis=0)
        print(X_LS_pairs.columns)
        print(importances)

        print("Done.")

    # ------------------------------- Tuning ------------------------------- #
    
    tune = True
    if tune == True:
        print("Tuning in progress...")
        
        X_tuning = load_from_csv('data/input_train_set.csv')
        y_tuning = load_from_csv('data/output_train_set.csv')
        
        models = [tree, knn, random_forest, gradient_boosting]
        models_names = ["tree", "knn", "random forest", "gradient_boosting"]
        tree_parameters = [1]
        knn_parameters = [1]
        forest_parameters = [50, 100, 150]
        gradient_boosting_parameters = [50, 100, 150, 200]

        # Trees
        parts_X, part_y = split_dataset(X_LS_pairs, y_LS_pairs, 5, shuffle=True)
        for p in tree_parameters:
            for i, (X, y) in enumerate(zip(parts_X, part_y)):
                pass

         
        print(f"The best model is {model_name} with parameter = {parameter}.")
        print(f"score = {score}")
    
    # ------------------------------- Learning ------------------------------- #
    
    learning = False
    if learning == True:
        tree_ = False
        if tree_ == False:
            print("Learning with a tree...")
            
            parameter = 8
            model = tree
            with measure_time("Tree learning"):
                clf = model(X_LS_pairs, y_LS_pairs, parameter)
            
            print("Done.")
            
            
        random_forest_ = True
        if random_forest_ == True:
            print("Learning with a random forest...")
            
            parameter = 105
            model = random_forest
            with measure_time("Random forest learning"):
                clf = model(X_LS_pairs, y_LS_pairs, parameter)
            
            print("Done.")
    
        gb = True
        if gb == True:
            print("Learning with a gb...")
            
            parameter = 200
            model = gradient_boosting
            with measure_time("Gradient boosting"):
                clf = model(X_LS_pairs, y_LS_pairs, parameter)
            
            print("Done.")

    # ------------------------------- Cross validation ------------------------------- #
    
    cv = False
    if cv is True:
        print("Evalution of the accuracy with cross validation...")
    
        k = 5
        with measure_time("Cross validation"):
            cv_score, cv_var = k_fold_cv_score(X_LS_pairs, y_LS_pairs, k, model, parameter)
        print(f"Cross validation score = {cv_score}, and variance = {cv_var}")
    
        print("Done.")  
    

    # ------------------------------ Prediction ------------------------------ #
    
    submission = False
    if submission is True:
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
