# %%
# problem 5.10: fit Y(x; a0, a1) = a0*x + a1*x^2 to trajectories from balle for y0=0. 
# Try different values for initial velocity but keep initial angle at 45deg
# include cases where air resistance is significant
# what is the largest velocity for which a parabola accurately fits the data?
# %%
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import Balle
# %%
x1, y1 = Balle.traj(0, 15, 45, 0.1, True)
# %%
fig,ax = plt.subplots()
ax.scatter(x1,y1)
# %%
p, result = np.polynomial.polynomial.polyfit(x1,y1,2,full=True)
pfit = np.polynomial.polynomial.polyval(x1,p)
resid = y1-pfit
print(result)
print(resid)
# %%
def fit_plot_traj(x, y):
    # do the polynomial fit
    p, result = np.polynomial.polynomial.polyfit(x,y, 2,full=True)
    pfit = np.polynomial.polynomial.polyval(x,p)
    resid = y-pfit
    print(p)
    # plot the results
    fig, ax = plt.subplots(2,1,sharex=True, gridspec_kw={'hspace': 0})
    ax[0].scatter(x, y, label = 'Original data')
    ax[0].plot(x, pfit, label = 'Fit')
    ax[1].scatter(x, resid)
    ax[1].axhline(0)
    ax[1].set_xlabel('x [m]')
    ax[0].set_ylabel('y [m]')
    ax[1].set_ylabel('Residuals [m]')   
    ax[1].legend()
# %%
x1, y1 = Balle.traj(y0=0, speed=15, theta=45.0, tau = 0.1, air_resist=True)
fit_plot_traj(x1,y1)
# %%
def f(x, a, b):
    return a*x**2 + b*x 

def residual(p, x, y):
    return y - f(x, *p)
# %%
p0 = [1., 1.]
p = spo.leastsq(residual, p0, args=(x1, y1)) 
print(p)
pfit = f(x1, *p[0])
# %%
def fit_plot_traj(x, y, constant=True):
    # do the polynomial fit
    if constant:
        p, result = np.polynomial.polynomial.polyfit(x,y, 2,full=True)
        pfit = np.polynomial.polynomial.polyval(x,p)
    else:
          # do the polynomial fit, without constant
        p0 = [1., 1.]
        p = spo.leastsq(residual, p0, args=(x, y)) 
        pfit = f(x, *p[0])
    resid = y-pfit
    print(p)
    # plot the results
    fig, ax = plt.subplots(2,1,sharex=True, gridspec_kw={'hspace': 0})
    ax[0].scatter(x, y, label = 'Original data')
    ax[0].plot(x, pfit, label = 'Fit')
    ax[1].scatter(x, resid)
    ax[1].axhline(0)
    ax[1].set_xlabel('x [m]')
    ax[0].set_ylabel('y [m]')
    ax[1].set_ylabel('Residuals [m]')   
    ax[1].legend()
# %%
x2,y2 = Balle.traj(y0=0, speed=50, theta=45.0, tau = 0.1, air_resist=True)
fit_plot_traj(x2,y2, constant=True)
# %%
