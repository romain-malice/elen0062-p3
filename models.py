import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 


def tree(X_LS_pairs, y_LS_pairs, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_LS_pairs, y_LS_pairs.values)
    return clf

def knn(X_LS_pairs, y_LS_pairs, n_neighbors):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_LS_pairs, y_LS_pairs.values)
    return clf

def svm(X_LS_pairs, y_LS_pairs, C):
    clf = SVC(C = C, probability=True)
    clf.fit(X_LS_pairs, y_LS_pairs.values)
    return clf
