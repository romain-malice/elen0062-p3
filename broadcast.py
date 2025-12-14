import numpy as np


example = 1

if example == 0:   
    
    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])
    
    result = a[None, :] + b[:, None]
    print(result)

if example == 1:    
    positions = np.array([[(1,1),(1,2),(1,3)],
                          [(1,1),(1,4),(1,5)],
                          [(1,1),(1,6),(1,10)]])
    
    diff = positions[:, :, None, :] - positions[:, None, :, :]
    
    distances = np.linalg.norm(diff, axis=-1)
    print(distances[0,0])
    print(np.delete(distances[0, 0], 0))
    
if example == 2:
    positions = np.array([[1,2,3],
                          [10,20,30],
                          [100,200,300]])
    summ = positions[:, None] + positions[None, :]
