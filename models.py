import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def test_some_ks_on_knn(ks, x, y, nb_splits):
    means = np.zeros_like(ks, dtype=float)
    variances = np.zeros_like(ks, dtype=float)
    for idx, k in enumerate(ks):
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, x, y, cv=nb_splits, scoring='accuracy')
        means[idx] = np.mean(scores)
        variances[idx] = np.var(scores)
    return means, variances

def test_trees(depths, x, y, nb_splits):
    means = np.zeros_like(depths, dtype=float)
    variances = np.zeros_like(depths, dtype=float)
    for idx, depth in enumerate(depths):
        clf = DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(clf, x, y, cv=nb_splits, scoring='accuracy')
        means[idx] = np.mean(scores)
        variances[idx] = np.var(scores)
    return means, variances
