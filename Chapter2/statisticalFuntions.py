import numpy as np

A = np.array([[1,2,3],[4,5,6]])
print('Array\n',A)
print('Sum all\n',np.sum(A))
print('Colum sums, summed along rows\n',np.sum(A,0)) # column sums, summed along rows
print('Row sums, summed along columns\n',np.sum(A,1)) # row sums, summed along the columns
print('\n')

print('Kertoma\n',np.prod(A))
print('Kertoma sarake\n',np.prod(A,0))
print('Kertoma rivi\n',np.prod(A,1))
print('\n')

print('Pienen elementti\n',np.min(A))
np.min(A,0)
np.min(A,1)
print('Suurin elementti\n',np.max(A))
np.max(A,0)
np.max(A,1)
print('\n')

print('Keskiarvo\n',np.mean(A))
np.mean(A,0)
np.mean(A,1)
print('\n')

print('Mediaani\n',np.median(A))
print('\n',np.median(A,0))
print('\n',np.median(A,1))

print(np.std(A))
np.std(A,0)
np.std(A,1)
print(np.var(A))
np.var(A,0)
np.var(A,1)