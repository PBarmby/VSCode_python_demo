import numpy as np
from timeit import Timer

li = list(range(500000))
nump_arr = np.array(li)
print(li[0],li[-1], nump_arr[0],nump_arr[-1])

def python_for():
    for num in li:
        num = num +1
    return ()

def numpy_add():
    return(nump_arr + 1)

print(min(Timer(python_for).repeat(10, 10)))
print(min(Timer(numpy_add).repeat(10, 10)))

