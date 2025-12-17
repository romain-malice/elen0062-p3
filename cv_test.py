import numpy as np
import pandas as pd


k = 3
block = 3

array = np.array([1,2,3,4,5,6,7,8,9,10])
df = pd.DataFrame(data=array, columns=['first'])

n = len(df)

# nombre total de lignes utilisables
usable_rows = (n // (k * block)) * (k * block) # // --> resultat arrondi vers le bas


# on tronque le dataframe si nécessaire
df_trimmed = df.iloc[:usable_rows]

# taille d'une part
part_size = usable_rows // k   # // --> pour avoir un int (car normalement 
                               # usable_rows est divisible par k)


# découpage en k parts
parts = [
    df_trimmed.iloc[i * part_size : (i + 1) * part_size]
    for i in range(k)
]

k = 1
x_to_fit = parts[:k] + parts[k+1:]   

# print(parts[k])
# print(parts[k].reset_index(drop=True))


x = pd.concat(x_to_fit, ignore_index=True)
print(x)

X_features_filtered = pd.concat(
        [])
