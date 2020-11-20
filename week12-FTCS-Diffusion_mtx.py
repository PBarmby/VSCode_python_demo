# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Adaptation of the `dftcs` program from the textbook into a function that will integrate the 1D diffusion equation in the FTCS approximation forward in time.

# %%
def diffusion_ftcs(nspace, ntime, tau_rel, args = [1.0, 1.0]):
    """ 
        Compute the solution to the diffusion equation using the forward-time, centered-space algorithm
        nspace: number of spatial grid points
        ntime: number of time grid points
        tau_rel: timestep in units of t_sigma=h**2/kappa
        args: list containing L and kappa
        
        output: tt(nspace, ntime): 2D ndarray containing T(x,t)
        
    """
    if tau_rel < 1.0 :
        print('Solution is expected to be stable')
    else:
        print('WARNING: Solution is expected to be unstable')
        
    L = args[0] # The system extends from x=-L/2 to x=L/2
    kappa = args[1] # Diffusion coefficient
    h = L/(nspace-1)   # Grid size
    t_sigma = h**2/(2*kappa) # critical time step
    tau = tau_rel * t_sigma
    coeff = kappa*tau/h**2

    #* Set initial and boundary conditions.
    tt = np.zeros((nspace, ntime))    # Initialize temperature to zero at all points
    tt[int(nspace/2),0] = 1./h             # Initial cond. is delta function in center
    ## The boundary conditions are tt[0,:] = tt[N-1, :] = 0

    for istep in range(1, ntime):  ## MAIN LOOP ##
    
        #* Compute new temperature using FTCS scheme.
        tt[1:(nspace-1), istep] = ( tt[1:(nspace-1), istep-1] + 
          coeff*( tt[2:nspace, istep-1] + tt[0:(nspace-2), istep-1] - 2*tt[1:(nspace-1), istep-1] ) )
        
    return(tt)

# %% [markdown]
# ### Part 2
# 
# Adaptation of the above function into a version that uses the matrix formulation.

# %%
def diffusion_ftcs_mtx(nspace, ntime, tau_rel, implicit=True, args = [1.0, 1.0]):
    """ 
        Compute the solution to the diffusion equation using the forward-time, centered-space algorithm in matrix formulation
        nspace: number of spatial grid points
        ntime: number of time grid points
        tau_rel: timestep in units of t_sigma=h**2/kappa
        args: list containing L and kappa
        
        output: tt(nspace, ntime): 2D ndarray containing T(x,t)
        
    """
    if tau_rel < 1.0 :
        print('Solution is expected to be stable')
    else:
        print('WARNING: Solution is expected to be unstable')
        
    L = args[0] # The system extends from x=-L/2 to x=L/2
    kappa = args[1] # Diffusion coefficient
    h = L/(nspace-1)   # Grid size
    t_sigma = h**2/(2*kappa) # critical time step
    tau = tau_rel * t_sigma
    coeff = kappa*tau/h**2

    #* Set initial and boundary conditions.
    tt = np.zeros((nspace, ntime))    # Initialize temperature to zero at all points
    tt[int(nspace/2),0] = 1./h             # Initial cond. is delta function in center

    # construct matrix D, NM4P p222
    D = -2*np.identity(nspace)+np.diagflat(np.ones(nspace-1),1)+np.diagflat(np.ones(nspace-1),-1)
    # boundary conditions
    D[0] = 0 
    D[-1] = 0
    
    if implicit:
        A = np.linalg.inv(np.identity(nspace) - coeff*D)
    else: # explicit FTCS 
        A = np.identity(nspace) + coeff*D 
    
    for istep in range(1, ntime):  ## MAIN LOOP: note it starts at 1, not zero: istep = 0 is the IC #
        #* Compute new temperature using FTCS scheme.
        tt[:, istep]  = A.dot(tt[:,istep-1])        
    return(tt)


# %%
# compute h from h = L/(N-1)
h=1/60.0
tsig = h**2/(2) # compute t_sigma
trel = 1e-4/tsig # compute trel
xplot = np.arange(61)*h - 0.5
tplot = np.arange(300)*(trel*tsig) 

# %%
tt = diffusion_ftcs(61, 300, trel)
tt_mtx = diffusion_ftcs_mtx(61, 300, trel)

# %%
print(np.abs(tt-tt_mtx).sum())


# %%
def doplot(xplot, tplot, tt, ptype):
    fig = plt.figure()
    if ptype == 'mesh':
        ax = fig.gca(projection = '3d')
        Tp, Xp = np.meshgrid(tplot, xplot)
        ax.plot_surface(Tp, Xp, tt, rstride=2, cstride=2, cmap=cm.gray)
        ax.set_xlabel('Time')
        ax.set_ylabel('x')
        ax.set_zlabel('T(x,t)')
        ax.set_title('Diffusion of a delta spike')
    elif ptype == 'contour':
        levels = np.linspace(0., 5., num=21) 
        ct = plt.contour(tplot, xplot, tt, levels) 
        plt.clabel(ct, fmt='%1.2f') 
        plt.xlabel('Time')
        plt.ylabel('x')
        plt.title('Temperature contour plot')      
    plt.show()
    return


# %%
doplot(xplot,tplot, tt, 'mesh')
doplot(xplot,tplot, tt_mtx, 'mesh')


# %%
doplot(xplot,tplot, tt, 'contour')
doplot(xplot,tplot, tt_mtx, 'contour')


# %%



