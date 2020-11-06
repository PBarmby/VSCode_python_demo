import numpy as np
import matplotlib.pyplot as plt

# implement equation 6.11
def tg(x, t, x0=0, kappa=1):
    """
    Return a normalized Gaussian of x, centered at x0
    with sigma = sqrt(2*kappa*t)
    """

    sigma = np.sqrt(2*kappa*t)
    prefactor = 1.0/(sigma*np.sqrt(2*np.pi))
    expl = np.exp(-(x-x0)**2/(2*sigma**2))
    return(prefactor*expl)

# implement equation 6.16
def T_analy(x, t, L=1):
    """
    returns analytical solution to diffusion eqn
    using method of images 
    """
    Nval = 50 # why 50? it seemed to work. 10 was also OK.
    Tsum = 0
    for n in range(-Nval, Nval): # sum the Gaussians 
        Tsum += (-1)**n * tg(x + n*L, t)
    return(Tsum)

# reproducing Figure 6.6, bottom panel
L = 1
xpts = np.arange(-1.5*L, 1.5*L, step=0.1)
T1 = T_analy(xpts, 0.03)

fig,ax = plt.subplots()
ax.plot(xpts, T1)
ax.set_xlabel('x/L')
ax.set_ylabel('T(x,t)')
plt.show()