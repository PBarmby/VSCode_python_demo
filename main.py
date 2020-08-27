import matplotlib.pyplot as plt
import numpy as np

xval = np.arange(0, np.pi*4, np.pi*4/50)
yval = np.cos(xval)

#print(xval, yval)

figure, ax = plt.subplots()
ax.scatter(xval,yval)
ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel(r'cos($\theta$)')
plt.show()
