import numpy as np

A = np.array([[2,1],[-4,3]])
b = np.array([11,3])
X = np.linalg.solve(A,b)

Ainv = np.linalg.inv(A)
print(np.matmul(Ainv,b))