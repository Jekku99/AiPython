import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_pd = pd.read_csv("quadreg_data.csv",skiprows=0,names=["x","y"])
print(data_pd)

xpd = np.array(data_pd[["x"]])
ypd = np.array(data_pd[["y"]])
xpd = xpd.reshape(-1,1)
ypd = ypd.reshape(-1,1)

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(xpd)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, ypd)

plt.scatter(xpd, ypd, color='red')
xval = np.linspace(-1,1,10).reshape(-1,1)
plt.plot(xval, pol_reg.predict(poly_reg.fit_transform(xval)), color='blue')
plt.show()

print(pol_reg.coef_)
print("c=",pol_reg.intercept_)