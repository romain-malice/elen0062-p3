import numpy as np
import pandas as pd


array = np.array([1,2,3,4,5,6,7,8,9,10])
df = pd.DataFrame(data=array, columns=['first'])

k=3
dfs = np.array_split(df, k)


dfs = [part.reset_index(drop=True) for part in dfs]



p=0.3
n = int(len(df) * p)
print(n)
df_30 = df.iloc[:n].reset_index(drop=True)
df_70 = df.iloc[n:].reset_index(drop=True)

print(df_70)