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

def k_fold_cv_score(X, y, k, training_model):
    block = 22
    n = X.shape[0]
    usable_rows = (n // (k * block)) * (k * block) 
    X_truncated = X.iloc[:usable_rows]
    y_truncated = y.iloc[:usable_rows]
    size = usable_rows // k  
    parts_X = [X_truncated.iloc[i * size : (i + 1) * size]
             for i in range(k)]
    parts_y = [y_truncated.iloc[i * size : (i + 1) * size]
             for i in range(k)]
    scores = np.zeros(k)
    for i in range(k):
        x_to_fit = parts_X[:k] + parts_X[k+1:]   
        x_to_fit = pd.concat(x_to_fit, ignore_index=True)

        y_to_fit = parts_y[:k] + parts_y[k+1:] 
        y_to_fit : pd.concat(y_to_fit, ignore_index=True)
        
        x_to_test = parts_X[i].reset_index(drop=True)
        y_to_test = parts_y[i].reset_index(drop=True)
        
        trained_model = training_model(x_to_fit, y_to_fit)
        #s'occuper du changement de formulation
        scores[i] = trained_model.s
    

    
    return

# def k_fold_cv_tree(k, depth, x_learn, y_learn):
#     """
#     Returns the cross validation error for a given number 
#     of subsets k, a given depth and a given learning datset.

#     """
#     x_subsets = np.array_split(x_learn, k)
#     y_subsets = np.array_split(y_learn, k)
#     for x, y, i in zip(x_subsets, y_subsets, range(k)):
#         x_to_fit = x_subsets.copy()
#         del x_to_fit[i]
#         y_to_fit = y_subsets.copy()
#         del y_to_fit[i]

#         tree = DecisionTreeClassifier(max_depth=depth)
#         tree.fit(np.concatenate(x_to_fit, axis=0),
#                  np.concatenate(y_to_fit, axis=0))
#         errors[i] = 1 - tree.score(x, y)

#     return np.mean(errors)





if __name__ == "__main__":
    pass
