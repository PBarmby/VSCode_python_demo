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
help(scint.solve_ivp)
# %%
# ODE dy/dt = -2 y between t = 0 and 4,
#  with the initial condition y(t=0) = 1
# %%
def dydt(t, y):
    return(-2.0*y)
# %%
result = scint.solve_ivp(dydt, (0,4), np.array([1.0]))
print(result)
# %%
%matplotlib inline
fig, ax = plt.subplots()
ax.plot(result.t, result.y[0])
# %%
print(result.t)
# %%
t_soln = np.linspace(0,4,100)
result = scint.solve_ivp(dydt, (0,4), np.array([1.0]), t_eval=t_soln)
print(result)
# %%
#The position of a mass on a spring obeys the 2nd order ODE 
# $y'' + 2 \varepsilon \omega_0 y' + \omega_0^2 y = 0$ 
# with $\omega_0^2 = k/m$ with $k$ the spring constant, 
# $m$ the mass and $\varepsilon = c/(2 m \omega_0)$ 
# with $c$ the damping coefficient.
# %%
# define some variables and compute some constants
mass = 0.5  # kg
kspring = 4  # N/m
cviscous = 0.4  # N s/m
eps = cviscous / (2 * mass * np.sqrt(kspring/mass))
omega = np.sqrt(kspring / mass)
# %%
# state vector [y, yprime]
# dydt function returns [yprime, yprimeprime]
def deriv_ystate(time, ystate, epsilon, omega_spring):
    yprime = ystate[1]
    y2prime = -2.0 * epsilon * omega_spring * ystate[1] - omega_spring **2 * ystate[0]
    return(yprime, y2prime)

# %%
time_soln = np.linspace(0,10,100)
yinit = (1,0)
result = scint.solve_ivp (deriv_ystate, (0,10), yinit, t_eval = t_soln, args = (eps, omega,))
# %%
print(result)
# %%
fig, ax = plt.subplots()
ax.plot(result.t, result.y[0])
# %%
fig, ax = plt.subplots()
ax.plot(result.t, result.y[1])
# %%
