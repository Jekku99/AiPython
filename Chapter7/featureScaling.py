import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv('weight-height.csv',skiprows=0,delimiter=",")
X = 2.54*df[['Height']]

X_mm = MinMaxScaler().fit_transform(X)
X_std = StandardScaler().fit_transform(X)

X = np.array(X)
X_mm = np.array(X_mm)
X_std = np.array(X_std)

plt.subplot(1,3,1)
plt.hist(X,30)
plt.xlabel("Height")
plt.title("original")
plt.subplot(1,3,2)
plt.hist(X_mm,30)
plt.title("Normalized")
plt.subplot(1,3,3)
plt.hist(X_std,30)
plt.title("Standardized")
plt.show()

X_mm2 = (X-np.min(X))/(np.max(X)-np.min(X))
print("diff=",np.max(np.abs(X_mm-X_mm2)))

X_std2 = (X-np.mean(X))/np.std(X)
print("diff2=",np.max(np.abs(X_std-X_std2)))