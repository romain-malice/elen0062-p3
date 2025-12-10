import numpy as np
import pandas as pd

from numpy.typing import NDArray

from file_interface import load_from_csv

def same_team_(sender,player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)

def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = np.array(["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team"])
    X_pairs = pd.DataFrame(data=np.zeros((n_*22,len(pair_feature_col))), columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_*22, 1)), columns=np.array(["pass"]))

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        sender = X_.iloc[i].sender_id
        players = np.arange(1, 23)
        #other_players = np.delete(players, sender-1)
        p_i_ = X_.iloc[i]
        for player_j in players:

            X_pairs.iloc[idx] = [sender,  p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)], same_team_(sender, player_j)]

            if not y_ is None:
                y_pairs.loc[idx, "pass"] = int(player_j == y_.iloc[i].values[0])
            idx += 1

    return X_pairs, y_pairs

def compute_distance_(X_):
    d = np.zeros((X_.shape[0],))

    d = np.sqrt((X_["x_sender"]-X_["x_j"])**2 + (X_["y_sender"]-X_["y_j"])**2)
    return d

def make_basic_features(x, y=None):
    X_LS_pairs, y_LS_pairs = make_pair_of_players(x, y)
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)

    return X_LS_pairs[["distance", "same_team"]], y_LS_pairs
