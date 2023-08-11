import numpy as np
x = np.array([1,3,4,5])
A = np.array([[1,3],[4,5]])

def väli():
    print('\n')
np.shape(x)
np.shape(A)
print("Zeros one dimensional")
Z = np.zeros(5)
print(Z)
np.shape(Z)
print("\n Zeros two dimensional" )
Z2 = np.zeros((4,5))
print(Z2)
np.shape(Z2)
print("\n Ones two dimensional")
Y = np.ones((2,3))
print(Y)
print("\n Full")
F = np.full((7,8),11)
print(F)
print('\n')

x = np.linspace(0,5,10)
print(x)
väli()

x2 = np.arange(0,5,0.2)
print(x2)
väli()

a=1
b=6
amount = 50
nopat = np.random.randint(a,b+1,amount)
print(nopat)
väli()

x = np.random.randn(100)
print(x)
väli()

x = np.random.random(10)
print(x)
väli()

x.size
x.ndim
A.size
A.ndim

A = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print(A)
väli()
np.shape(A)
print(A)
väli()
A.reshape(3,4)
print(A)
väli()
A.reshape(2,3,2)
print(A)
väli()
A.reshape(6,2)
print(A)
väli()

A = np.repeat([[1,2,3]],4,axis=0)
print(A)
väli()
B = np.repeat([[1],[2],[3]],3,axis=1)
print(B)
väli()

A = np.array([1,2])
B = A.copy()
B[0] = 99
print(B)
print(A)

A = np.array([[1,2,3],[4,5,6]])
print('shape',np.shape(A))

print(np.linspace(0,5,6))
print(np.arange(0,5,1))

A = np.array ([[1,2,3], [4,5,6]])
print(A[1,2])
