import numpy as np
import matplotlib.pyplot as plt

positions = np.random.rand(3, 10, 2) * 10
senders = [3, 1, 7]
angles = np.zeros([len(senders), 10])
# Iterate over all passes

diff = positions[:, :, None, :] - positions[:, None, :, :]
distances = np.linalg.norm(diff, axis=-1)   

idx0= 0
idx1=0
for i, sender in enumerate(senders):
    x_s, y_s = positions[i, sender, :]
    x_r, y_r = np.split(positions[i, :, :], 2, axis=1)

    delta_y = y_r - y_s # (nb_passes x nb_players)
    delta_x = x_r - x_s # "

    angles[i] = np.arctan2(delta_y, delta_x).T

view_angles = np.zeros_like(angles)

for i in range(3):
    sort_indices = np.argsort(angles[i])
    inverse_indices = np.argsort(sort_indices)

    sorted_angles = angles[i][sort_indices]
    continuous_sorted_angles = np.concatenate(([sorted_angles[-1] - 2*np.pi], sorted_angles, [sorted_angles[0] + 2*np.pi]))
    view_angle_wrong_idx = continuous_sorted_angles[2:] - continuous_sorted_angles[:-2]
    print(view_angle_wrong_idx)
    view_angles[i] = view_angle_wrong_idx[inverse_indices]
    print(view_angles)

plt.scatter(positions[0, :, 0], positions[0, :, 1])
for i, pos in enumerate(positions[0, :, :]):
    plt.annotate("{}".format(i), pos)

print(angles[0])
print(view_angles[0] * 180/np.pi)

plt.show()
