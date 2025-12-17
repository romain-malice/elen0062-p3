import numpy as np
import pandas as pd


k = 3
block = 3

a = np.array([2,1, 11,4,52,87,7,8,9,10])
players_i = np.arange(10)

sort = np.argsort(a)
inv = np.argsort(sort)

a_sorted = a[sort]

b_sorted = a[a > 10]



print(a)
print(a_sorted)
print(b)
