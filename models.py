import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


def tree(X_LS_pairs, y_LS_pairs, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_LS_pairs, y_LS_pairs)
    return clf

def knn(X_LS_pairs, y_LS_pairs, n_neighbors):
    return

def svm(X_LS_pairs, y_LS_pairs, parameter):
    return
