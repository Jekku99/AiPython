import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("weight-height.csv",delimiter=",",skip_header=1)
lenght = data[:,1]*2.54
weight = data[:,2]*0.45359237

print('Mean lenght: ',np.mean(lenght))
print('Median lenght: ',np.median(lenght))
print('Standard lenght deviation: ',np.std(lenght))
print('Lenght variance: ',np.var(lenght))
print()
print('Mean weight: ',np.mean(weight))
print('Median weight: ',np.median(weight))
print('Standard weight deviation: ',np.std(weight))
print('Weight variance: ',np.var(weight))

plt.hist(lenght,50)
plt.title('Lenghts')
plt.xlabel('cm')
plt.show()