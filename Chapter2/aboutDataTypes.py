import numpy as np

A = np.array([1,2,3,4,5,6],dtype="int8")

print(A.dtype)

A.astype("int16")

print(A.dtype)

print(A.itemsize)
print(A.nbytes)
A2 = np.array([1,2,3,4,5,6],dtype="int64")
print(A2.nbytes)

A = np.array([1,2,3,4,5,6],dtype="uint8")
print(A-7)