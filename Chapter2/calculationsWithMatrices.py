import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([[3,9],[4,-1]])
print('A\n',A)
print('B\n',B,'\n')

print('+\n',A+B)
print('-\n',A-B)
print('*\n',A*B)
print('/\n',A/B)
print('\n')

T = 5*np.ones((10,10))
print(T)

print('A-1\n',A-1,'\n')
print('B+2\n',B+2,'\n')

print('A**2\n',A**2,'\n')

print('Matrix multiplication\n',np.matmul(A,B)) # or A @ B

b = np.array([[5],[7]])
print(np.matmul(A,b))

A = np.array([1,2,3,4,5,6,7])
B = A[A>3]

print('B\n', B)

A = np.array([[2,1],[-4,3]])
b = np.array([11,3])
X = np.linalg.solve(A,b)

print('Solve X', X)