import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.tree import DecisionTreeClassifier     # tree


def proba_to_player(y_proba, nb_passes):
    good_shape = y_proba[:, 1].reshape(nb_passes, 22) 
    return np.argmax(good_shape, axis=1) + 1


def accuracy(y, y_pred):
    # Check if size matches
    nb_samples = len(y)
    if y_pred.size != nb_samples:
        raise ValueError("y_true and y_prediction msut have the same size")
    
    # Compute number of right predictions
    nb_right_predictions = (y == y_pred).sum()

    # Compute accuracy
    return (nb_right_predictions / nb_samples)

def brier_score(y_true: NDArray[np.int64], p_predict: NDArray[np.float64]) -> float:
    # Number of observations
    nb_obs = y_true.size

    # Create y_ij
    p_ideal = np.zeros([nb_obs, 22])
    for idx, y in enumerate(y_true):
        p_ideal[idx, y - 1] = 1

    # Compute score
    return ((p_ideal - p_predict)**2).sum()

if __name__ == "__main__":
    pass
