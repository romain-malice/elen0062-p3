import numpy as np
import pandas as pd

from features import make_features
from evaluation import accuracy, k_fold_cv_score

from models import tree, knn, random_forest

def data_process(X, y):
    n = int(0.7 * X.shape[0])
    
    X_train = X.iloc[:n].reset_index(drop=True)
    y_train = y.iloc[:n].reset_index(drop=True)
    
    X_test = X.iloc[n:].reset_index(drop=True)  
    y_test = y.iloc[n:].reset_index(drop=True)
    
    X_train_pairs, y_train_pairs = make_features(X_train, y_train)
    X_test_pairs, y_test_pairs = make_features(X_test, y_test)
    
    return X_train_pairs, y_train_pairs, X_test_pairs, y_test_pairs
    

def tuning_tree(X, y, parameters):
    X_train_pairs, y_train_pairs, X_test_pairs, y_test_pairs = data_process(X, y)
        
    k = 5
    scores = [k_fold_cv_score(X_train_pairs, y_train_pairs, k, tree, p)
              for p in parameters]
    best_parameter = parameters[np.argmax(scores)]
    
    clf_test = tree(X_train_pairs, y_train_pairs, best_parameter)
    score_TS = accuracy(clf_test, X_test_pairs, y_test_pairs)
    
    return best_parameter, score_TS

def tuning_knn(X, y, parameters):
    X_train_pairs, y_train_pairs, X_test_pairs, y_test_pairs = data_process(X, y)
        
    k = 5
    scores = [k_fold_cv_score(X_train_pairs, y_train_pairs, k, knn, p)
              for p in parameters]
    best_parameter = parameters[np.argmax(scores)]
    
    clf_test = knn(X_train_pairs, y_train_pairs, best_parameter)
    score_TS = accuracy(clf_test, X_test_pairs, y_test_pairs)
    
    return best_parameter, score_TS 

def tuning_forest(X, y, parameters):
    X_train_pairs, y_train_pairs, X_test_pairs, y_test_pairs = data_process(X, y)
        
    k = 3
    scores = [k_fold_cv_score(X_train_pairs, y_train_pairs, k, random_forest, p)
              for p in parameters]
    best_parameter = parameters[np.argmax(scores)]
    
    clf_test = random_forest(X_train_pairs, y_train_pairs, best_parameter)
    score_TS = accuracy(clf_test, X_test_pairs, y_test_pairs)
    
    return best_parameter, score_TS 


def tuning(X_tuning, y_tuning, models, models_names, tree_parameters, knn_parameters, forest_parameters):
    parts_X = np.array_split(X_tuning, 4)
    parts_y = np.array_split(y_tuning, 4)
    
    X_train = pd.concat(parts_X[:3], ignore_index=True)
    y_train = pd.concat(parts_y[:3], ignore_index=True)
    X_test = parts_X[3].reset_index(drop=True)
    y_test = parts_y[3].reset_index(drop=True)
    
    print("Finding best parameters for tree...")
    tree_parameter, tree_score = tuning_tree(parts_X[0].reset_index(drop=True),
                                             parts_y[0].reset_index(drop=True), tree_parameters)
    print(f"Best parameter for tree is {tree_parameter}")
    print("Done.")
    print("Finding best parameters for knn...")
    knn_parameter, knn_score = tuning_knn(parts_X[1].reset_index(drop=True), 
                                          parts_y[1].reset_index(drop=True), knn_parameters)
    print(f"Best parameter for knn is {knn_parameter}")

    print("Done.")
    print("Finding best parameters for random forest...")
    forest_parameter, forest_score = tuning_forest(parts_X[2].reset_index(drop=True), 
                                     parts_y[2].reset_index(drop=True), forest_parameters)
    print(f"Best parameter for random forest is {forest_parameter}")

    print("Done.")
    
    
    print("Finding best model....")
    
    parameters = [tree_parameter, knn_parameter, forest_parameter]
    scores = [tree_score, knn_score, forest_score]
    idx_best = np.argmax(scores)
    best_model = models[idx_best]
    best_model_name = models_names[idx_best]
    best_parameter = parameters[idx_best]   
    
    print(models_names)
    print(scores)
    
    print("Done.")


    print("Evaluating the best model..")
    
    X_train_pairs, y_train_pairs = make_features(X_train, y_train)
    X_test_pairs, y_test_pairs = make_features(X_test, y_test)
    
   
    clf = best_model(X_train_pairs, y_train_pairs, best_parameter)
    score = accuracy(clf, X_test_pairs, y_test_pairs)

    print("Done.")    
    
    return best_model_name, best_parameter, score
