import numpy as np
import pandas as pd
from numpy.typing import NDArray


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

def brier_score(y_true: NDArray[np.int64], p_predict: NDArray[np.float64]) -> float:
    # Number of observations
    nb_obs = y_true.size

    # Create y_ij
    p_ideal = np.zeros([nb_obs, 22])
    for idx, y in enumerate(y_true):
        p_ideal[idx, y - 1] = 1

    # Compute score
    return ((p_ideal - p_predict)**2).sum()

def split_dataset(X_pairs, y_pairs, k, shuffle=False):
    block = 22
    n = X_pairs.shape[0]
    if shuffle is True:
        perm = np.random.permutation(len(X_pairs))
        X_pairs = X_pairs.loc[perm].reset_index(drop=True)
        y_pairs = y_pairs.loc[perm].reset_index(drop=True)
    usable_rows = (n // (k * block)) * (k * block) 
    X_truncated = X_pairs.iloc[:usable_rows]
    y_truncated = y_pairs.iloc[:usable_rows]
    size = usable_rows // k  
    parts_X = [X_truncated.iloc[i * size : (i + 1) * size]
             for i in range(k)]
    parts_y = [y_truncated.iloc[i * size : (i + 1) * size]
             for i in range(k)]
    return parts_X, parts_y


def k_fold_cv_score(X_pairs, y_pairs, k, training_model, parameter):
    print(f"cross validation with parameter = {parameter} ")
    block = 22
    n = X_pairs.shape[0]
    usable_rows = (n // (k * block)) * (k * block) 
    X_truncated = X_pairs.iloc[:usable_rows]
    y_truncated = y_pairs.iloc[:usable_rows]
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
      
        trained_model = training_model(x_to_fit, y_to_fit, parameter)
        
        scores[i] = accuracy(trained_model, x_to_test, y_to_test)
    
    return np.mean(scores), np.var(scores)


if __name__ == "__main__":
    pass
