import enum
import numpy as np
import pandas as pd
import time
import os

def same_team_(sender,player_j):
    if sender <= 11:
        return player_j <= 11
    else:
        return player_j > 11

def positions_array(X_):
    n = X_.shape[0]
    positions = np.zeros((n, 22, 2))
    for j in range(22):
        positions[:, j, 0] = X_[f"x_{j+1}"].values
        positions[:, j, 1] = X_[f"y_{j+1}"].values
    return positions
    
def distances_array(X_):
    positions = positions_array(X_)
    diff = positions[:, :, None, :] - positions[:, None, :, :]
    distances = np.linalg.norm(diff, axis=-1)   
    return distances

def distances_to_goal(X_):
    # TODO

    positions = positions_array()
    return

def angle_array(X):
    positions = positions_array(X)
    senders = X["sender_id"].values
    angles = np.zeros([len(senders), 22])
    # Iterate over all passes
    for i, sender in enumerate(senders):
        x_s, y_s = positions[i, sender, :]
        x_r, y_r = np.split(positions[i, :, :], 2, axis=1)

        delta_y = y_r - y_s # (nb_passes x nb_players)
        delta_x = x_r - x_s # "
    
        angles[i] = np.arctan2(delta_y, delta_x).T

    return angles

def make_features(X_, y_=None):
    nb_passes = X_.shape[0]
    distances = distances_array(X_)
    columns_name  = ["sender", "receiver","same team", "dist_s_r", "d_min_s_opp", "d_min_s_teammate", 
                     "d_min_r_opp", "d_min_r_teammate"]
    X_pairs = pd.DataFrame(data=np.zeros((nb_passes*22, len(columns_name))), columns=columns_name)
    y_pairs = pd.DataFrame(data=np.zeros((nb_passes*22,1)), columns=np.array(["pass"]))
    idx = 0
    for i, pass_idx in enumerate(X_.index):
        sender = X_.loc[pass_idx, "sender_id"] - 1
        if sender <= 10:
            teammates = np.arange(0, 11)
            opponents = np.arange(11, 22)
        else:
            teammates = np.arange(11, 22)
            opponents = np.arange(0, 11)

        for receiver in range(0, 22):
            same_team = same_team_(sender, receiver)

            dist_s_r = distances[i, sender, receiver]
            d_min_s_opp = min(distances[i, sender, opponents])
            d_min_s_teammate = min(distances[i, sender, teammates[teammates != sender]])

            if same_team is True:
                d_min_r_opp = min(distances[i, receiver, opponents])
                d_min_r_teammate = min(distances[i, receiver, teammates[teammates != receiver]])
            else:
                d_min_r_opp = min(distances[i, receiver, teammates])
                d_min_r_teammate = min(distances[i, receiver, opponents[opponents != receiver]])
            
            X_pairs.iloc[idx] = [sender + 1, receiver + 1, int(same_team), dist_s_r, d_min_s_opp, d_min_s_teammate,
                                  d_min_r_opp, d_min_r_teammate]
            if not y_ is None:
                y_pairs.loc[idx, "pass"] = int(receiver == y_.loc[pass_idx].values[0] - 1)
            idx += 1

    return X_pairs, y_pairs

def write_features_file(features, name):
    file_name = '{}.csv'.format(name)
    features.to_csv(path_or_buf=os.path.join("features", file_name))
