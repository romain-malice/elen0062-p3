import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier




def k_fold_cv_tree(k, depth, x_learn, y_learn):
    """
    Returns the cross validation error for a given number 
    of subsets k, a given depth and a given learning datset.

    """
    errors = np.zeros(k)
    x_subsets = np.array_split(x_learn, k)
    y_subsets = np.array_split(y_learn, k)
    for x, y, i in zip(x_subsets, y_subsets, range(k)):
        x_to_fit = x_subsets.copy()
        del x_to_fit[i]
        y_to_fit = y_subsets.copy()
        del y_to_fit[i]

        tree = DecisionTreeClassifier(max_depth=depth)
        tree.fit(np.concatenate(x_to_fit, axis=0),
                 np.concatenate(y_to_fit, axis=0))
        errors[i] = 1 - tree.score(x, y)

    return np.mean(errors)


def k_fold_cv_knn(k, n_neighbors, x_learn, y_learn):
    """
     Returns the cross validation error for a given number 
     of subsets k, a given number of neighbors and a given learning datset.

    """
    errors = np.zeros(k)
    x_subsets = np.array_split(x_learn, k)
    y_subsets = np.array_split(y_learn, k)
    for x, y, i in zip(x_subsets, y_subsets, range(k)):
        x_to_fit = x_subsets.copy()
        del x_to_fit[i]
        y_to_fit = y_subsets.copy()
        del y_to_fit[i]

        x_to_fit = np.concatenate(x_to_fit, axis=0)
        y_to_fit = np.concatenate(y_to_fit, axis=0)

        if n_neighbors >= len(y_to_fit):
            n_neighbors = len(y_to_fit) - 1
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_to_fit, y_to_fit)
        errors[i] = 1 - knn.score(x, y)

    return np.mean(errors)
