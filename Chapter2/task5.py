import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,7,100)
y=2*x+1
plt.plot(x,y,'g',linestyle="--")
y=2*x+2
plt.plot(x,y,'b',linestyle=":")
y=2*x+3
plt.plot(x,y,'r',linestyle="-.")
plt.title('Otsikko')
plt.xlabel('x')
plt.ylabel('y')
plt.show()