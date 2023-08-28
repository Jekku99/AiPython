import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv',skiprows=0,delimiter=",")
X = df[['Height']]
y = df[['Weight']]

plt.scatter(X, y)
plt.show()

regr = linear_model.LinearRegression()
regr.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

poly_reg = PolynomialFeatures(degree=1)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test,color="red")
xval = np.linspace(54,80,10).reshape(-1,1)
plt.plot(xval, pol_reg.predict(poly_reg.fit_transform(xval)), color='black')
plt.legend(["train","test"])
plt.xlabel("Height Inch")
plt.ylabel("Weight Lbs")
plt.show()

yhat = regr.predict(X)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, yhat)))
print('R2 value:', metrics.r2_score(y, yhat))