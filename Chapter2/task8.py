import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([[-1,1],[5,7]])
AB = np.matmul(A,B)

print(np.linalg.det(A))
print(np.linalg.det(B))
print(np.linalg.det(AB))
