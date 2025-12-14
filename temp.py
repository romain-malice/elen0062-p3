import pandas as pd
import numpy as np

from file_interface import load_from_csv

X = load_from_csv("data/input_train_set.csv")
y = load_from_csv("data/output_train_set.csv")

print(y.iloc[0].values)
