# %%
import fibo
# %%
fibo.fib(42)
# %%
fibo.fib2(42)
# %%
result = fibo.fib2(42)
print(result)
# %%
result.append(47)
# %%
print(result)
# %%
help(fib)
# %%
help(fibo.fib)
# %%
import fibo
# %%
help(fibo.fib)
# %%
import importlib
# %%
importlib.reload(fibo)
# %%
help(fibo.fib)
# %%
help(fibo.fib2)
# %%
fibo.fib2(33)
# %%
importlib.reload(fibo)
# %%
import sys
print(sys.path)
# %%
dir(sys)
# %%
help(sys.exit)
# %%
dir(fibo)
# %%
print(cos(3.14))
# %%
print(np.cos(3.14))
# %%
import numpy as np
import math as mt
# %%
print(np.cos(3.14))
# %%
print(mt.cos(3.14))
# %%
help(cos)
# %%
help(mt.cos)
# %%
help(np.cos)
# %%
testarr = np.zeros(5)
# %%
np.cos(testarr)
# %%
print(np.cos(testarr))
# %%
print(mt.cos(testarr))
# %%
# DONT DO THIS!
# from numpy import *
# %%
# numpy and scipy overlaps:
#  linalg
#  random
#  stats
# %%
dir(np)
# %%
import scipy as sp
# %%
dir(sp)
# %%
