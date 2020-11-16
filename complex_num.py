# %%
import cmath
import numpy as np
# %%
x = 4+3j
# %%
type(x)
# %%
j =7
# %%
print(j,type(j))
# %%
for j in range(0,10):
    x = j+1j
    print(x)
# %%
print(x**2)
# %%
print(1j**2)
# %%
x.real
# %%
print(x,x.real,x.imag)
# %%
cmath.polar(x)
# %%
cmath.exp(x)
# %%
cmath.exp(0+cmath.pi*1j)
# %%
cmath.exp(0+cmath.pi*j)
# %%
print(j)
# %%
a = np.array([1+2j, 3+4j, 5+6j])
# %%
np.real(a)
# %%
np.imag(a)
# %%
np.conj(a)
# %%
a = np.zeros((3,3),dtype='complex')
# %%
print(a)
# %%
a[0,0]=1+3j
# %%
np.angle([1.0, 1.0j, 1+1j], deg=True) 
# %%
np.absolute([1.0, 1.0j, 1+1j])
# %%
type(a[0,0])
# %%
