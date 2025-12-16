import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def tree(X_LS_pairs, y_LS_pairs, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_LS_pairs, y_LS_pairs.values.ravel())
    return clf

def knn(X_LS_pairs, y_LS_pairs, n_neighbors):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_LS_pairs, y_LS_pairs.values.ravel())
    return clf

def random_forest(X_LS_pairs, y_LS_pairs, parameter):
    max_depth = 9
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=parameter)
    clf.fit(X_LS_pairs, y_LS_pairs.values.ravel())
    return clf
