import numpy as np

A = np.array([[1,2,3],[4,5,6]])
print(A[0,0])
print(A[0,1])

A[:,0] # fisrt column, ":"reads all rows
A[0,:] # first row, ":" reads all columns
print('\n Array')
A = np.array([[1,2,3],[4,5,6]])
A[0,0] = 17
A[1,:] = [11,12,13]
print(A)
print('\n vertical stack')
new = np.vstack((A,A))
print(new)
print('\n horisontal stack')
new2 = np.hstack((A,A))
print(new2)
print('\n')
print('Array')
A = np.array([1,2,3,4,5,6])
A = A.reshape(2,3)
n,m = np.shape(A)
print('rows')
for i in range(n):
    print("Row",i,"is",A[i,:])
print('columns')
for j in range(m): 
    print("Column",j,"is",A[:,j])
print('alkio')
for i in range(n):
    for j in range(m):
        print("Element",i,j,"is",A[i,j])
print('alkiot')
for a in np.nditer(A):
    print(a)