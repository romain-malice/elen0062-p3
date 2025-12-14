import pandas as pd
import numpy as np

from file_interface import load_from_csv
from features import same_team_

X = load_from_csv("data/input_train_set.csv")
y = load_from_csv("data/output_train_set.csv")

nb_failed = 0
for sender, receiver in zip(X['sender_id'], y['receiver_id']):
    if not same_team_(sender, receiver):
        nb_failed += 1

print(nb_failed/len(y))
