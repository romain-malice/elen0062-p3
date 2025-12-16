import numpy as np
import pandas as pd

from features import make_features
from evaluation import accuracy, k_fold_cv_score

from models import tree

def tuning_tree(X, y):
    parameters = [5, 10, 15]
    n = int(0.7 * X.shape[0])
    X_to_k_fold = X.iloc[:n].reset_index(drop=True)
    y_to_k_fold = y.iloc[:n].reset_index(drop=True)
    
    X_test = X.iloc[n:].reset_index(drop=True)  
    y_test = y.iloc[n:].reset_index(drop=True)
    
    X_pairs, y_pairs = make_features(X_to_k_fold, y_to_k_fold) 
    k = 5
    scores = [k_fold_cv_score(X_pairs, y_pairs, k, tree, p)
              for p in parameters]
    
    return parameters[np.argmax(scores)]

def tuning_knn(X, y):
    return 

def tuning_svm(X, y):
    return 

def tuning_model(tree_parameter, knn_parameter, svm_parameter):
    
    return 
def tuning(X_tuning, y_tuning):
    parts_X = np.array_split(X_tuning, 4)
    parts_X = [parts_X.reset_index(drop=True) for part in parts_X]
    
    parts_y = np.array_split(y_tuning, 4)
    parts_y = [parts_y.reset_index(drop=True) for part in parts_y]
    
    print("Finding best parameters...")
    tree_parameter = tuning_tree(parts_X[0], parts_y[0])
    knn_parameter = tuning_knn(parts_X[1], parts_y[1])
    svm_parameter = tuning_svm(parts_X[2], parts_y[2])
    print("Done.")
    
    print("Finding best model....")
    
    best_model, best_parameter = tuning_model(tree_parameter, knn_parameter, svm_parameter)
    
    print("Done.")
    
    return best_model, best_parameter