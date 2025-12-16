import numpy as np
import matplotlib.pyplot as plt

positions = np.random.rand(3, 10, 2) * 10
senders = [3, 1, 7]
angles = np.zeros([len(senders), 10])

diff = positions[:, :, None, :] - positions[:, None, :, :]
distances = np.linalg.norm(diff, axis=-1)   

teammates = np.arange(5)
opps = np.arange(5, 10)

for i in range(3):
    print(positions[i, teammates].mean(axis=0))
    print(positions[i, opps].mean(axis=0))
