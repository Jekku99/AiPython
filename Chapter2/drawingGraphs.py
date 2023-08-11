import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
x = [1,2,3,4]
y = [1,4,9,16]
plt.plot(x,y)
plt.plot(x,y,"o")
plt.plot(x,y,linewidth=2,linestyle="-.")
plt.show()

x = np.linspace(0,7,100)
y = np.sin(x)
plt.plot(x,y)
plt.plot(x,y,"g")
plt.show()

plt.title('Title')
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,y,color="#4b0082")
plt.plot(x,y,"D")
plt.plot(x,y,"go",x,2*y,"r^")
plt.legend(['sin(x)','cos(x)'])
plt.show()

plt.subplot(1,2,1) # 1x2 grid 1. subpicture
plt.plot(x,y)
plt.title("first")
plt.subplot(1,2,2) # 1x2 grid 2. subpicture
plt.plot(x,2*y)
plt.title("second")
plt.suptitle("Common Title")
plt.show()

plt.bar(['2018','2019','2020'],[120000,125000,130000],color="blue")
plt.title("Title")
plt.xlabel("years")
plt.ylabel("Sales")
plt.show()

x = np.random.randn(2000)
plt.hist(x,10)
plt.ylabel('frequencies')
#plt.scatter(x,y,color="r",marker="o",label="Points")
plt.show()

points = np.arange(-2,2,0.01)
x,y = np.meshgrid(points,points)
z = np.sqrt(x**2 + y**2)
plt.imshow(z)
plt.colorbar()
plt.show()

x, y = np.meshgrid(np.linspace(-2, 2, 30),np.linspace(-2, 2, 30))
z = np.cos(x ** 2 + y ** 2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z)
ax.set_title('Texture')
plt.savefig('picture.pdf')
plt.show()
