import pandas as pd

data = pd.read_csv('dataset.csv', header=None, index_col=None)
for i in range(len(data)):
    data.iloc[i, -1] = data.iloc[i, -1] - 1
print(data)

data.to_csv("new_dataset.csv", header=None)
