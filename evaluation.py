import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.tree import DecisionTreeClassifier     # tree


def accuracy(y_true: NDArray[np.int64], y_prediction: NDArray[np.int64]) -> float:
    # Check if size matches
    nb_samples = y_true.size
    if y_prediction.size != nb_samples:
        raise ValueError("y_true and y_prediction msut have the same size")
    
    # Compute number of right predictions
    nb_right_predictions = (y_true == y_prediction).sum()

    # Compute accuracy
    return nb_right_predictions / nb_samples

def brier_score(y_true: NDArray[np.int64], p_predict: NDArray[np.float64]) -> float:
    # Number of observations
    nb_obs = y_true.size

    # Create y_ij
    p_ideal = np.zeros([nb_obs, 22])
    for idx, y in enumerate(y_true):
        p_ideal[idx, y - 1] = 1

    # Compute score
    return ((p_ideal - p_predict)**2).sum()


def make_model(model, parameter):
    if model == 'knn':
        return KNeighborsClassifier(n_neighbors=parameter)
    if model == 'tree':
        return DecisionTreeClassifier(max_depth=parameter)
    else:
        return


def k_fold_cv(k, parameter, model, x_learn, y_learn):
    """
    Returns the cross validation error for a given number 
    of subsets k, a given parameter for the considered model
    and a given learning dataset.

    """
    errors = np.zeros(k)
    x_subsets = np.array_split(x_learn, k)
    y_subsets = np.array_split(y_learn, k)
    for x, y, i in zip(x_subsets, y_subsets, range(k)):
        x_to_fit = x_subsets.copy()
        del x_to_fit[i]
        y_to_fit = y_subsets.copy()
        del y_to_fit[i]
        
        x_to_fit = pd.concat(x_to_fit, axis=0, ignore_index=True)
        y_to_fit = pd.concat(y_to_fit, axis=0, ignore_index=True)
        
        if model == 'knn':
            if parameter >= len(y_to_fit):
                parameter = len(y_to_fit) - 1
        
        model = make_model(model, parameter)
        if hasattr(model, "fit") and hasattr(model, "score"):
            model.fit(x_to_fit, y_to_fit)
            errors[i] = 1 - model.score(x, y)
        else:
            raise(TypeError(f"Wrong argument type for `model`. Expected a sklearn classifier but got {type(model)}"))
    return np.mean(errors), np.var(errors)
