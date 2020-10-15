# %%
import numpy.linalg as np_linalg
import scipy.linalg as sp_linalg
import numpy as np
# %%
# Ax = b
A = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,2],
])
b= np.array([1,0,1])
# %%
for i in range (A.shape[0]):
    terms = []
    for j in range(A.shape[1]):
        terms.append("{1} x[{0}]".format(j, A[j,i]))
    print(" + ".join(terms), "=", b[i])

# %%
x = np_linalg.solve(A,b)
print(x)
# %%
np.dot(A,x) -b
# %%
xnew = sp_linalg.inv(A).dot(b)
print(xnew)
# %%
Ainv = sp_linalg.inv(A)
# %%
A.dot(Ainv)
# %%
Ainv.dot(A)
# %%
A.dot(np.linalg.inv(A))
# %%
# A x_i = lambda_i x_i
# I omega_i = lambda_i omega_i 
Isquare = np.array([[2/3, -1/4], [-1/4, 2/3]]) 
lambdas, omegas = np_linalg.eig(Isquare)
# %%
print(lambdas, omegas)
# %%
# eigenvectors are omegas[:,i]
# transpose 
omegas.T
# %%
# test: (I- lambdas*1)omegas = 0 ?
(Isquare - lambdas[0]*np.identity(2)).dot(omegas[:,0])
# %%
(Isquare - lambdas[0]*np.identity(2)).dot(omegas.T[0])
# %%
(Isquare - lambdas[1]*np.identity(2)).dot(omegas.T[1])
# %%
sigma_y = np.array([[0,-1j],[1j,0]])
E, chis = np.linalg.eig(sigma_y)
print(E)
print(chis.T)
# %%
chi1 = chis.T[0]
print(chi1)
norm = np.dot(chi1.conjugate(),chi1)
chi1hat = chi1/np.sqrt(norm)
print(chi1hat)
# %%
