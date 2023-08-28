import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('Admission_Predict.csv',skiprows=0,delimiter=",")

print(df.head())

X = df[['CGPA']]
y = df[['Chance of Admit ']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test,color="red")
plt.legend(["train","test"])
plt.xlabel("CGPA")
plt.ylabel("Chance of Admit")
plt.title("Dataset splitting")
plt.show()

print(X.head())
print(X_train.head())
print(X_test.head())

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
plt.scatter(X,y)
plt.plot(X_test,lm.predict(X_test),color="red")
plt.title("Prediction")
plt.show()

print("R2=",lm.score(X_test,y_test)) # using linear model
print("R2=",metrics.r2_score(y_test,predictions))# using sklearn metrics