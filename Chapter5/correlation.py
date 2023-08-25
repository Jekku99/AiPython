import pandas as pd
df = pd.read_csv("weight-height.csv",usecols=[1,2])
print(df.corr())