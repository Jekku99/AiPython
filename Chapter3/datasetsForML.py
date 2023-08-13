import pandas as pd

df = pd.read_csv("iris.csv")
print(df.head())
print(df.tail())
print(df.describe())

print(df.dtypes)
print(df.index)
print(df.columns)
print(df.values)

df2 = df.sort_values('sepal_width',ascending=False) # does not sort in-place
print(df2)

print(df[['sepal_width']]) # slice one column by name
print(df[['sepal_width','sepal_length']]) # slice two columns by name

print(df[2:4]) # slice rows by index, exclusive

print(df.loc[2:4,['petal_width','petal_length']]) # slice rows by index and columns by name
print(df.iloc[2:4,[0,1]]) # slice row and columns by index

print(df[df.sepal_width>3]) # slicing with logical condition
print(df[df['species'].isin(["Iris-setosa"])])

df['sepal_area'] = df.sepal_length*df.sepal_width
print(df)
df['zeros'] = 0.0
print(df)

df = df.drop(['zeros'],axis=1)
print("df after drop",df)

df.rename(columns = {'sepal_area':'sep_ar'},inplace=True)
print(df.head())
df.columns = ['col1','col2','col3','col4','col5','col6']
print(df.head())

#to_append = [7.0,4.0,5.5,6.6,"Iris-setosa",28.0]
#a_series = pd.Series(to_append, index = df.columns)
#df = df.append(a_series, ignore_index=True)
#print(df)

for ind, row in df.iterrows():
    print(ind,row['col2'])

df.to_csv("iris_new.csv")