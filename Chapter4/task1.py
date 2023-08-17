import numpy as np
import matplotlib.pyplot as plt

loops = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]
low=1
high=7
for n in loops:
    n1 = np.random.randint(low, high, n)
    n2 = np.random.randint(low, high, n)
    s = n1 + n2

    h,h2 = np.histogram(s,range(2,14))
    plt.bar(h2[:-1],h/n)
    plt.title(n)
    plt.show()