import pandas as pd

d = {'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data=d)

part_of_df = df[1:3].copy()

df = df.drop(index=[1, 2])

print(df)
print(part_of_df)
