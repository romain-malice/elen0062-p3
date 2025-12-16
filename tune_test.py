import numpy as np
import pandas as pd

example = 0
array = np.array([1,2,3,4,5,6,7,8,9,10])
df = pd.DataFrame(data=array, columns=['first'])
  
if example == 0:  
    k=3
    dfs = np.array_split(df, k)
    
    print(dfs[:2])
    dfs_ = pd.concat(dfs[:2], ignore_index=True)
    print(dfs_)
    dfs = dfs[2].reset_index(drop=True)
    print(dfs)
    

    


if example == 1:
    p=0.3
    n = int(len(df) * p)
    print(n)
    df_30 = df.iloc[:n].reset_index(drop=True)
    df_70 = df.iloc[n:].reset_index(drop=True)
    
    print(df_70)


