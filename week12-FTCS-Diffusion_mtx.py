# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Adaptation of the `dftcs` program from the textbook 
# into a function that will integrate the 1D diffusion 
# equation in the FTCS approximation forward in time.

# %%
def diffusion_ftcs(nspace, ntime, tau_rel, args = [1.0, 1.0]):
    """ 
        Compute the solution to the diffusion equation using 
          the forward-time, centered-space algorithm
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
    tt = np.zeros((nspace, ntime))    # Initialize T to zero at all points
    tt[int(nspace/2),0] = 1./h        # IC: delta function in center
    ## The boundary conditions are tt[0,:] = tt[N-1, :] = 0

    for istep in range(1, ntime):  ## MAIN LOOP ##
    
        #* Compute new temperature using FTCS scheme.
        tt[1:(nspace-1), istep] = ( tt[1:(nspace-1), istep-1] + 
          coeff*( tt[2:nspace, istep-1] + tt[0:(nspace-2), istep-1] - 2*tt[1:(nspace-1), istep-1] ) )
        
    return(tt)

# %% [markdown]
# ### Part 2
# 
# Adaptation of the above function into a version that 
# uses the matrix formulation.

# %%
def diffusion_ftcs_mtx(nspace, ntime, tau_rel, implicit=True, args = [1.0, 1.0]):
    """ 
        Compute the solution to the diffusion equation using 
            the forward-time, centered-space algorithm 
            in matrix formulation
        nspace: number of spatial grid points
        ntime: number of time grid points
        tau_rel: timestep in units of t_sigma=h**2/kappa
        implicit: use the implicit formulation (True) or
                      the explicit formulation (False)
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
    tt = np.zeros((nspace, ntime))    # Initialize T to zero at all points
    tt[int(nspace/2),0] = 1./h        # IC: delta function in center

    # construct matrix D, NM4P p222
    D = -2*np.identity(nspace) +np.diagflat(np.ones(nspace-1),1) +np.diagflat(np.ones(nspace-1),-1)
    # boundary conditions
    D[0] = 0 
    D[-1] = 0
    
    if implicit:
        A = np.linalg.inv(np.identity(nspace) - coeff*D)
    else: # explicit FTCS 
        A = np.identity(nspace) + coeff*D 

    ## MAIN LOOP: note it starts at 1, not zero: istep = 0 is the IC 
    for istep in range(1, ntime):  
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
tt = diffusion_ftcs(61, 300, trel)
tt_mtx_implicit = diffusion_ftcs_mtx(61, 300, trel, implicit=True)
tt_mtx_explicit = diffusion_ftcs_mtx(61, 300, trel, implicit=False)
diff = np.abs(tt_mtx_implicit-tt_mtx_explicit)

# %%
diff2 = np.abs(tt-tt_mtx_explicit)
print(diff2.sum()) 
# %%
doplot(xplot,tplot, tt_mtx_implicit, 'mesh')
doplot(xplot,tplot, tt_mtx_explicit, 'mesh')


# %%
doplot(xplot,tplot, tt_mtx_implicit, 'contour')
doplot(xplot,tplot, tt_mtx_explicit, 'contour')


# %%
plt.imshow(diff[20:40,0:10])



# %%
np.where(diff>1)
# %%
diff[np.where(diff>1)]
# %%
def diffusion_ftcs_mtx_v2(nspace, ntime, tau_rel, scheme='crank', args = [1.0, 1.0]):
    """ 
        Compute the solution to the diffusion equation using 
            the forward-time, centered-space algorithm 
            in matrix formulation
        nspace: number of spatial grid points
        ntime: number of time grid points
        tau_rel: timestep in units of t_sigma=h**2/kappa
        scheme: 'implicit', 'explicit', 'crank'
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
    tt = np.zeros((nspace, ntime))    # Initialize T to zero at all points
    tt[int(nspace/2),0] = 1./h        # IC: delta function in center

    # construct matrix D, NM4P p222
    D = -2*np.identity(nspace) +np.diagflat(np.ones(nspace-1),1) +np.diagflat(np.ones(nspace-1),-1)
    # boundary conditions
    D[0] = 0 
    D[-1] = 0
    
    if scheme == 'implicit':
        A = np.linalg.inv(np.identity(nspace) - coeff*D)
    elif scheme== 'explicit': # explicit FTCS 
        A = np.identity(nspace) + coeff*D 
    elif scheme == 'crank': 
        A = np.matmul(np.linalg.inv(np.identity(nspace) - coeff*D),
            np.identity(nspace) + coeff*D)

    ## MAIN LOOP: note it starts at 1, not zero: istep = 0 is the IC 
    for istep in range(1, ntime):  
        #* Compute new temperature using FTCS scheme.
        tt[:, istep]  = A.dot(tt[:,istep-1])        
    return(tt)

# %%
tt_mtx_cn = diffusion_ftcs_mtx_v2(61, 300, trel, scheme='crank')
diff = np.abs(tt_mtx_implicit-tt_mtx_cn)
# %%
print(diff.sum())
# %%
doplot(xplot,tplot, tt_mtx_implicit, 'contour')
doplot(xplot,tplot, tt_mtx_explicit, 'contour')
doplot(xplot,tplot, tt_mtx_cn, 'contour')

# %%
plt.imshow(diff[20:40,0:50])
# %%
