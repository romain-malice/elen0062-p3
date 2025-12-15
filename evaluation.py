import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.tree import DecisionTreeClassifier     # tree


def proba_to_player(proba_pairs):
    nb_passes = proba_pairs.shape[0] // 22
    proba_passes = proba_pairs[:, 1].reshape(nb_passes, 22) 
    player_predictions  = np.argmax(proba_passes, axis=1) + 1
    return player_predictions, proba_passes


def accuracy(clf, X_TS_pairs, y_TS_pairs):
    nb_passes = X_TS_pairs.shape[0] // 22
    y_TS_pairs = y_TS_pairs["pass"].values
    player_target = np.argmax(y_TS_pairs.reshape(nb_passes, 22), axis=1) + 1

    proba_pairs = clf.predict_proba(X_TS_pairs)
    player_predictions, _ = proba_to_player(proba_pairs)

    nb_right_predictions = (player_predictions == player_target).sum()

    return (nb_right_predictions / nb_passes)

def brier_score(y_true, p_predict):
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
        x_to_fit = parts_X[:i] + parts_X[i+1:]   
        x_to_fit = pd.concat(x_to_fit, ignore_index=True)

        y_to_fit = parts_y[:i] + parts_y[i+1:] 
        y_to_fit = pd.concat(y_to_fit, ignore_index=True)
        
        x_to_test = parts_X[i].reset_index(drop=True)
        y_to_test = parts_y[i].reset_index(drop=True)
      
        trained_model = training_model(x_to_fit, y_to_fit)
        
        scores[i] = accuracy(trained_model, x_to_test, y_to_test)
    
    return np.mean(scores)


if __name__ == "__main__":
    pass
