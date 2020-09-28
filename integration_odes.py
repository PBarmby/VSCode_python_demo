# %% 
# %%
import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt
# %%
help(np.trapz)
# %%
x = np.arange(0,10)
y = x**2
int_result= np.trapz(y,x)
print(int_result, 10**3/3)
# %%
x = np.linspace(0,10,100)
y = x**2
int_result= np.trapz(y,x)
print(int_result, 10**3/3)
# %%
help(scint)
# %%
int_result = scint.romb(y,x)
print(int_result, 10**3/3)
# %%
x = np.linspace(0,10,65)
y = x**2
int_result = scint.romb(y,x)
print(int_result, 10**3/3)
# %%
help(scint.romb)
# %%
dx = x[1]-x[0]
# %%
int_result = scint.romb(y,dx)
print(int_result, 10**3/3)
# %%
help(scint.quad)
# %%
def myfunc(x):
    return(x**2 + np.sin(x))
# %%
result = scint.quad(myfunc(1,4))
print(result)
# %%
result = scint.quad(myfunc,1,4)
print(result)
# %%
def myfunc2(x, a=1):
    return(x**2 + a*np.sin(x))
# %%
result = scint.quad(myfunc2,1,4, args=(1,))
print(result)
# %%
result = scint.quad(myfunc2,1,4, args=(0,))
print(result)
# %%
def myfunc3(x):
    return(3*np.exp(x))
scint.quad(myfunc3,0,np.inf)
# %%
