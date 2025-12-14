import numpy as np
import pandas as pd


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
    positions = positions_array()

    return


def make_features(X_, y_=None):
    nb_passes = X_.shape[0]
    distances = distances_array(X_)
    columns_name  = ["sender", "receiver","same team", "dist_s_r", "d_min_s_opp", "d_min_s_teammate", 
                     "d_min_r_opp", "d_min_r_teammate"]
    X_pairs = pd.DataFrame(data=np.zeros((nb_passes*22, len(columns_name))), columns=columns_name)
    y_pairs = pd.DataFrame(data=np.zeros((nb_passes*22,1)), columns=np.array(["pass"]))
    idx = 0
    for i in range(nb_passes):
        sender = X_.at[i, 'sender_id'] - 1
        if sender <= 10:
            teammates = np.arange(0, 11)
            opponents = np.arange(11, 22)
        else:
            teammates = np.arange(11, 22)
            opponents = np.arange(0, 11)

        for receiver in teammates:
            dist_s_r = distances[i, sender, receiver]
            d_min_s_opp = min(distances[i, sender, opponents])
            d_min_s_teammate = min(distances[i, sender, teammates[teammates != sender]])
            d_min_r_opp = min(distances[i, receiver, opponents])
            d_min_r_teammate = min(distances[i, receiver, teammates[teammates != receiver]])
            
            X_pairs.iloc[idx] = [sender + 1, receiver + 1, 1, dist_s_r, d_min_s_opp, d_min_s_teammate,
                                  d_min_r_opp, d_min_r_teammate]
            if not y_ is None:
                y_pairs.loc[idx, "pass"] = int(receiver == y_.iloc[i].values[0] - 1)
            idx += 1
            
        for receiver in opponents:
            dist_s_r = distances[i, sender, receiver]
            d_min_s_opp = min(distances[i, sender, opponents])
            d_min_s_teammate = min(distances[i, sender, teammates[teammates != sender]])
            d_min_r_opp = min(distances[i, receiver, teammates])
            d_min_r_teammate = min(distances[i, receiver, opponents[opponents != receiver]])
            
            X_pairs.iloc[idx] = [sender + 1, receiver + 1, 0, dist_s_r, d_min_s_opp, d_min_s_teammate,
                                  d_min_r_opp, d_min_r_teammate]
            if not y_ is None:
                y_pairs.loc[idx, "pass"] = int(receiver == y_.iloc[i].values[0] - 1)
            idx += 1           
               
    return X_pairs, y_pairs
