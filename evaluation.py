import numpy as np
from numpy.typing import NDArray


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
