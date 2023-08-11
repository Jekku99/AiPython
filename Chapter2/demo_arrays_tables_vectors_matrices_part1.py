#
# Examples for Tables, Vectors, Matrices, Graphs
#
import numpy as np

A = np.array([1,2,3,244,249091,1221])
print(A)

B = np.array([[1,2,44],[3,4,441]])
print(B)

print(np.shape(A))
print(np.shape(B))

############################
# zeros(), ones() and full()
############################

Z = np.zeros((15))
print("Z=",Z)
Z2 = np.zeros( (4,7)   )
print("Z2=",Z2)

I = np.ones((6,2))
print("I=",I)

F = np.full((3,4),2.5)
print("F=",F)


#linspace() and arange()
x = np.linspace(5,10,11)
print("x=",x)

y = np.arange(5,10,0.5)
print("y=",y)

# For comparison: range()
r = range(0,5,1)
for i in r:
    print(i)


# Random in numpy
# dice = np.random.random_integers(1,6,100) # this fuction will no longer be supported, use DeprecationWarning
dice = np.random.randint(1,7,30)
print("nopat=", dice)

xn = np.random.randn(100)
print("xn=",xn)

x1 = np.random.random(25)
print("x1=",x1)

# ndim ja size
print("B.ndim=",B.ndim)
print("B.size=",B.size)

# Reading file (CSV) (Comma Separated Values)
data = np.genfromtxt("data.csv",delimiter=",",skip_header=1)
print("data=",data)

#  Reshape()
A = np.linspace(0,11,12)
print(A.shape)
B = A.reshape(2,6)
print("B=",B)
C = A.reshape(3,4)
print("C=",C)

# Repeat()
row = [[1,2,3]]
A = np.repeat(row,40,axis=0)
print(A)
column = np.array([[1],[2],[3]])
B = np.repeat(column,5,axis=1)
print(B)


# Cutting/indexing Arrays
A = np.array([[1,2,3],[4,5,6]])
print("A=",A)
print(A[0,0])
print(A[0,1])
print(A[1,0])
print(A[0,:])
print(A[:,1])

x = np.linspace(10,21,12)
print("x=",x)
print(x[3:5])
print(x[2:8:2])

# Updating array by cutting and pasting
x[1] = 100
print("x=",x)
A[1,1] = 99
print("A=",A)
A[1,:] = [97,98,99]
print("A=",A)
A[1,:] = 101
print("A=",A)


# Repeat
A2 = np.vstack((A,A))
print("A2=",A2)
A3 = np.hstack((A,A))
print("A3=",A3)

# Delete()
dlt = np.delete(A,[0],axis=0)
print("pois=",dlt)
dlt2 = np.delete(A,[0],axis=1)
print("pois2=",dlt2)

# Iteration
A = np.array([1,2,3,4,5,6])
A = A.reshape(2,3)
print("A=",A)
n, m = np.shape(A)
print("n=",n,"m=",m)
for i in range(n):
    print("Row",i,"is",A[i,:])
for j in range(m):
    print("Column", j, "is", A[:, j])

for i in range(n):
    for j in range(m):
        print("Element", i,j, "is", A[i, j])

for a in np.nditer(A):
    print("a=",a)
