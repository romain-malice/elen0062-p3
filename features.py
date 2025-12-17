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

def angle_array(X):
    """
    Computes angular position of each player w.r.t. the sender
    """
    positions = positions_array(X)
    senders = X["sender_id"].values - 1
    angles = np.zeros([len(senders), 22])
    # Iterate over all passes
    for i, sender in enumerate(senders):
        x_s, y_s = positions[i, sender, :]
        x_r, y_r = np.split(positions[i, :, :], 2, axis=1)

        delta_y = y_r - y_s # (nb_passes x nb_players)
        delta_x = x_r - x_s # "
    
        angles[i] = np.arctan2(delta_y, delta_x).T

    return angles

def distances_to_goals(positions, player):
    left_goal, right_goal = (- 5250, 0), (5250, 0)
    left_player = np.argmin(positions[:, 0]) # Closest player to the left border
    right_player = np.argmax(positions[:, 0]) # Closest player to the right border
    if same_team_(left_player, right_player):
        # If the players are in the same team, look which one is the most centered along y
        # This is probably the keeper because he tries to protect its goal
        if abs(positions[left_player, 1]) < abs(positions[right_player, 1]): # Left player is the goal keeper
            if same_team_(left_player, player):
                d_team_goal = np.linalg.norm(left_goal - positions[left_player])
                d_opp_goal = np.linalg.norm(right_goal - positions[left_player])
            else:
                d_opp_goal = np.linalg.norm(left_goal - positions[left_player])
                d_team_goal = np.linalg.norm(right_goal - positions[left_player])
        else:
            if same_team_(right_player, player):
                d_opp_goal = np.linalg.norm(left_goal - positions[left_player])
                d_team_goal = np.linalg.norm(right_goal - positions[left_player])
            else:
                d_team_goal = np.linalg.norm(left_goal - positions[left_player])
                d_opp_goal = np.linalg.norm(right_goal - positions[left_player])
    else:
        if same_team_(right_player, player):
            d_opp_goal = np.linalg.norm(left_goal - positions[left_player])
            d_team_goal = np.linalg.norm(right_goal - positions[left_player])
        else:
            d_team_goal = np.linalg.norm(left_goal - positions[left_player])
            d_opp_goal = np.linalg.norm(right_goal - positions[left_player])
    return d_team_goal, d_opp_goal

def make_features(X_, y_=None):
    nb_passes = X_.shape[0]
    positions = positions_array(X_)
    distances = distances_array(X_)
    angles = angle_array(X_)
    columns_name  = ["pass_i", "sender", "x_s", "y_s", "receiver", "x_r", "y_r", "same team", "dist_s_r", "d_min_s_opp", "d_min_s_teammate", 
                     "d_min_r_opp", "d_min_r_teammate", "r_demarcation", "d_cm_team_s", "d_cm_team_r", "d_cm_opp_s",
                     "d_cm_opps_r", "d_team_goal_s", "d_team_goal_r", "d_opp_goal_s", "d_opp_goal_r"]
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

        # Center of mass
        cm_team = positions[i, teammates].mean(axis=0)
        cm_opp = positions[i, teammates].mean(axis=0)

        d_cm_team_s = np.linalg.norm(positions[i, sender] - cm_team)
        d_cm_opp_s = np.linalg.norm(positions[i, sender] - cm_opp)

        # Distance to goal
        d_team_goal_s, d_opp_goal_s = distances_to_goals(positions[i], sender)

        # Player visibility
        sort_indices = np.argsort(angles[i])
        inverse_indices = np.argsort(sort_indices)

        sorted_angles = angles[i, sort_indices]

        for receiver in range(0, 22):
            same_team = same_team_(sender, receiver)

            d_team = np.linalg.norm(positions[i, receiver] - cm_team)
            d_opp = np.linalg.norm(positions[i, receiver] - cm_opp)
            d_cm_team_r, d_cm_opp_r = (d_team, d_opp) if same_team else (d_opp, d_team)

            dist_s_r = distances[i, sender, receiver]
            d_min_s_opp = min(distances[i, sender, opponents])
            d_min_s_teammate = min(distances[i, sender, teammates[teammates != sender]])
            
            d_team_goal_r, d_opp_goal_r = distances_to_goals(positions[i], receiver)

            if sender == receiver:
                view_angle = 0.0
            else:
                mask = distances[i, sender, sort_indices] <= 1.5 * distances[i, sender, receiver]
                print(distances[i, sender, sort_indices])
                print(1.2 * distances[i, sender, receiver])
                print(mask)
                close_players_angles = sorted_angles[mask]
                continuous_sorted_angles = np.concatenate(([close_players_angles[-1] - 2*np.pi], close_players_angles, [close_players_angles[0] + 2*np.pi]))
                new_receiver_idx = mask[:inverse_indices[receiver]].sum()
                view_angle = continuous_sorted_angles[new_receiver_idx + 2] - continuous_sorted_angles[new_receiver_idx]

            if same_team is True:
                d_min_r_opp = min(distances[i, receiver, opponents])
                d_min_r_teammate = min(distances[i, receiver, teammates[teammates != receiver]])
            else:
                d_min_r_opp = min(distances[i, receiver, teammates])
                d_min_r_teammate = min(distances[i, receiver, opponents[opponents != receiver]])
            
            X_pairs.iloc[idx] = [pass_idx, sender + 1, positions[i, sender, 0], positions[i, sender, 1], 
                                 receiver + 1, positions[i, receiver, 0], positions[i, receiver, 1], 
                                 int(same_team), dist_s_r, d_min_s_opp, d_min_s_teammate,
                                 d_min_r_opp, d_min_r_teammate, view_angle, d_cm_team_s, d_cm_team_r, d_cm_opp_s,
                                 d_cm_opp_r, d_team_goal_s, d_team_goal_r, d_opp_goal_s, d_opp_goal_r]
            if not y_ is None:
                y_pairs.loc[idx, "pass"] = int(receiver == y_.loc[pass_idx].values[0] - 1)
            idx += 1

    return X_pairs, y_pairs

def write_features_file(features, name):
    file_name = '{}.csv'.format(name)
    features.to_csv(path_or_buf=os.path.join("features", file_name))
