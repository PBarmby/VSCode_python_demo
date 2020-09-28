# %%
import numpy as np
A = [[1,2], [3,4]]
B = np.array(A)
# %%
print A+A
# %%
print(A+A)
# %%
print(B+B)
# %%
print(B)
# %%
print(A-A)
# %%
print(B-B)
# %%
print(2*A)
# %%
print(2*B)
# %%
print(B*B)
# %%
print(A*A)
# %%
np.dot(B,B)
# %%
print(type(A),type(B), type(472.2), type('hello'))
# %%
A[0][0]
# %%
print(A,B)
# %%
B[0][0]
# %%
B[0,0]
# %%
A[0]
# %%
B[0]
# %%
B[0,:]
# %%
B[:,0]
# %%
A[:,0]
# %%
C = np.array([B,B])
# %%
C.shape
# %%
np.zeros((2,2))
# %%
np.ones((3,3))
# %%
y=np.linspace(0,20,11)
# %%
print(y)
# %%
x = y[:4]
print(x)
# %%
y[4]
# %%
print(y[2:10:2])
# %%
np.linalg.eig(B)
# %%
np.dot(B[0],B[0])# %%

# %%
for num in range(0,10):
    if num % 2 == 0:
        print(num)
    else:
        print('odd number!')
# %%
for item in A:
    print(item)
# %%
x, y = np.loadtxt('earthpop.dat', unpack = True)
# %%
import matplotlib.pyplot as plt
%matplotlib inline
# %%
fig, ax = plt.subplots()
plt.scatter(x,y, label='data')
plt.plot(x, testmodel)
ax.set_xlabel('time')
ax.set_ylabel('pop')
ax.set_ylim(0,y[-1])
ax.legend()
# %%
testmodel = np.exp(x-1650) + y[0]
# %%
print(np.mean(y), np.max(x))
# %%
print(y.mean(), x.max())
# %%
# scipy functions that operate on arrays: interpolating
from scipy.interpolate import interp1d
linear_interp = interp1d(x, y)
cubic_interp = interp1d(x, y, kind = 'cubic')
# %%
fig, ax = plt.subplots()
plt.scatter(x,y, label= 'data')
xnew = np.linspace(x[0],x[-1],200)
plt.plot(xnew, linear_interp(xnew), label = 'linear')
plt.plot(xnew, cubic_interp(xnew), label = 'cubic')
ax.set_xlabel('time')
ax.set_ylabel('pop')
ax.set_ylim(0,y[-1])
ax.legend()
# %%
# scipy functions that operate on functions: finding the roots of a function
import scipy.optimize as spo
def f(x):
    return x**2 + 10*np.sin(x)
fn_root = spo.root(f, x0=1) 
print(fn_root)
# %%
