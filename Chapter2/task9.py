import numpy as np

A = np.array([[1,2,3],[0,1,4],[5,6,0]])
Ainv = np.linalg.inv(A)

AA1 = np.matmul(A,Ainv)
A1A = np.matmul(Ainv,A)

print(AA1)
print()
print(A1A)