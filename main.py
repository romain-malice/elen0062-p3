# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, KFold

from file_interface import load_from_csv, write_submission

from features import make_features, write_features_file
from evaluation import proba_to_player, accuracy, k_fold_cv_score, split_dataset, k_fold_sets
from tuning import tuning

from models import *

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
    
    feature_selection = False
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
        # 

        print("Done.")

    # ------------------------------- Tuning ------------------------------- #
    
    tune = False
    tune2 = True
    if tune == True:
        print("Tuning in progress...")

        models = [tree, knn, svm_lin, svm_rbf, random_forest, gradient_boosting]
        
        # Testing 5 parameters for each model
        max_depths = [5, 10, 25, 50, 100]
        nb_neigh = [1, 11, 21, 31, 51]
        c_svm = [0.01, 0.1, 1, 50, 100]
        kernel_svm = ['linear', 'rbf']
        nb_trees_forest = [10, 50, 100, 150, 200]
        nb_trees_gb = [100, 150, 200, 250, 300]

        models_param = [max_depths, nb_neigh, c_svm, c_svm, nb_trees_forest, nb_trees_gb]

        # Initializing scores
        tree_scores = np.zeros([25, 5])
        knn_scores = np.zeros([25, 5])
        svm_lin_scores = np.zeros([25, 5])
        svm_rbf_scores = np.zeros([25, 5])
        forest_scores = np.zeros([25, 5])
        boost_scores = np.zeros([25, 5])

        best_model_scores = np.zeros(5)

        k = 5
        X_learn_out, y_learn_out, X_test_out, y_test_out = k_fold_sets(X_LS_pairs, y_LS_pairs, k, shuffle=True)
        for i, (Xlo, ylo, Xto, yto) in enumerate(zip(X_learn_out, y_learn_out, X_test_out, y_test_out)):
            # Outer loop -> model evaluation
            print(f"Starting {i + 1}th outer fold...")

            X_learn_in, y_learn_in, X_test_in, y_test_in = k_fold_sets(Xlo, ylo, k, shuffle=True)
            for j, (Xli, yli, Xti, yti) in enumerate(zip(X_learn_in, y_learn_in, X_test_in, y_test_in)):
                # Inner loop -> model selection
                print(f"Starting {j + 1}th inner fold...")

                # Test all models

                # Testing trees
                print("Learning trees...")
                trees = [DecisionTreeClassifier(max_depth=d) for d in max_depths]
                for l, tree in enumerate(trees):
                    with measure_time("tree"):
                        tree.fit(Xli, yli.values.ravel())
                        tree_scores[k * i + j, l] = accuracy(tree, Xti, yti)
                print("Done")

                # Testing knn
                print("Learning knn...")
                knns = [KNeighborsClassifier(n_neighbors=neigh) for neigh in nb_neigh]
                for l, knn in enumerate(knns):
                    with measure_time("knn"):
                        knn.fit(Xli, yli.values.ravel())
                        knn_scores[k * i + j, l] = accuracy(knn, Xti, yti)
                print("Done")

                # Testing svms
                #print("Learning linear svm...")
                #svm_lin = [SVC(kernel='linear', C=c, verbose=True) for c in c_svm]
                #for l, sl in enumerate(svm_lin):
                #    with measure_time("linear svm"):
                #        sl.fit(Xli, yli.values.ravel())
                #        svm_lin_scores[k * i + j, l] = accuracy(sl, Xti, yti)
                #print("Done")

                #print("Learning rbf svm...")
                #svm_rbf = [SVC(kernel='rbf', C=c, verbose=True) for c in c_svm]
                #for l, sr in enumerate(svm_rbf):
                #    with measure_time("rbf svm"):
                #        sr.fit(Xli, yli.values.ravel())
                #        svm_rbf_scores[k * i + j, l] = accuracy(sr, Xti, yti)
                #print("Done")

                # Testing random forests
                print("Learning forests...")
                forests = [RandomForestClassifier(n_estimators=nt, verbose=True, n_jobs=8) for nt in nb_trees_forest]
                for l, rf in enumerate(forests):
                    with measure_time("forest"):
                        rf.fit(Xli, yli.values.ravel())
                        forest_scores[k * i + j, l] = accuracy(rf, Xti, yti)
                print("Done")

                # Testing gradient boosting
                print("Learning boosters...")
                boosters = [GradientBoostingClassifier(n_estimators=nt, verbose=1) for nt in nb_trees_gb]
                for l, gb in enumerate(boosters):
                    with measure_time("gradient boosting"):
                        gb.fit(Xli, yli.values.ravel())
                        boost_scores[k*i + j, l] = accuracy(gb, Xti, yti)
                print("Done")

            # i-th inner k-fold done -> find the best model

            tree_s = np.mean(tree_scores[k*i:k*(i+1)], axis=0)
            knn_s = np.mean(knn_scores[k*i:k*(i+1)], axis=0)
            sl_s = np.mean(svm_lin_scores[k*i:k*(i+1)], axis=0)
            sr_s = np.mean(svm_rbf_scores[k*i:k*(i+1)], axis=0)
            forest_s = np.mean(forest_scores[k*i:k*(i+1)], axis=0)
            boost_s = np.mean(boost_scores[k*i:k*(i+1)], axis=0)
            
            mean_scores = np.array([tree_s, knn_s, sl_s, sr_s, forest_s, boost_s])
            best_model_idx, best_param_idx = np.unravel_index(np.argmax(mean_scores), mean_scores.shape)

            # Train the best model for that outer fold
            best_model = models[best_model_idx](Xlo, ylo, models_param[best_model_idx][best_param_idx])
            # Test the model
            best_model_scores[i] = accuracy(best_model, Xto, yto)
            print(f"Best performances on {i + 1}th fold:\
                  {models[best_model_idx].__name__} with parameter value of\
                  {models_param[best_model_idx][best_param_idx]} scored {best_model_scores[i]}% accuracy")

        # Out of the loop
        print(f"Final score: {np.mean(best_model_scores)}")
        pd.DataFrame(tree_scores, columns=pd.Index(max_depths)).to_csv("tuning/tree.csv")
        pd.DataFrame(knn_scores, columns=pd.Index(max_depths)).to_csv("tuning/knn.csv")
        pd.DataFrame(forest_scores, columns=pd.Index(max_depths)).to_csv("tuning/forest.csv")
        pd.DataFrame(boost_scores, columns=pd.Index(max_depths)).to_csv("tuning/gradient_boosting.csv")
    
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
