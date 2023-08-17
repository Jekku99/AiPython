import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("linreg_data.csv",skiprows=0,names=["x","y"])
print(data)

xpd = data["x"]
ypd = data["y"]
n = xpd.size
plt.scatter(xpd,ypd)
plt.show()

xbar = np.mean(xpd)
ybar = np.mean(ypd)

term1 = np.sum(xpd*ypd)
term2 = np.sum(xpd**2)

b = (term1-n*xbar*ybar)/(term2-n*xbar*xbar)
a = ybar - b*xbar

print('a = ',a)
print('b = ',b)

x = np.linspace(0,2,100)
y = a+b*x
plt.plot(x,y,color="black")
plt.scatter(xpd,ypd)
plt.scatter(xbar,ybar,color="red")
plt.show()

xval = 0.50
yval = a+b*xval

print(yval)

xval = np.array([0.5,0.75,0.90])
yval = a+b*xval
print(yval)

yhat = a+b*xpd
RSS = np.sum((ypd-yhat)**2)
print("RSS =",RSS)
RMSE = np.sqrt(np.sum((ypd-yhat)**2)/n)
print("RMSE=",RMSE)
MAE = np.sum(np.abs(ypd-yhat))/n
print("MAE =",MAE)
MSE = np.sum((ypd-yhat)**2)/n
print("MSE =",MSE)
R2 = 1-np.sum((ypd-yhat)**2)/np.sum((ypd-ybar)**2)
print("R2  =",R2)