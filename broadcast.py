import numpy as np

a = np.array([1, 2, 3])      # forme (3,)
b = np.array([[10], [20], [30]])  # forme (3,1)

result1 = a + b
print(result1)


vertical = np.array([1, 2, 3])
horizontal = np.array([10, 20, 30])

result2 = vertical[None, :] + horizontal[:, None]
print(result2)