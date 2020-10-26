# ftdemo - Discrete Fourier transform demonstration program
# updated to be compatible with Python3, 20201026 - PB

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt
import time

def time_fft(N, freq, phase, method):
    """Calculate a Fourier transform of a sine wave time series
    N: number of data points
    freq: frequency of sine wave
    phase: phase in radians
    method: 1 Direct summation, 2 FFT

    Output: time taken to compute
    """
    #* Initialize the sine wave time series to be transformed.

    tau = 1.   # Time increment
    t = np.arange(N)*tau               # t = [0, tau, 2*tau, ... ]
    y = np.empty(N)
    for i in range(N):              # Sine wave time series
        y[i] = np.sin(2*np.pi*t[i]*freq + phase)    
    f = np.arange(N)/(N*tau)           # f = [0, 1/(N*tau), ... ] 

    #* Compute the transform using desired method: direct summation
    #  or fast Fourier transform (FFT) algorithm.
    yt = np.zeros(N,dtype=complex)

    startTime = time.time()
    if int(method) == 1 :             # Direct summation
        twoPiN = -2. * np.pi * (1j) /N    # (1j) = sqrt(-1)
        for k in range(N):
            for j in range(N):
                expTerm = np.exp( twoPiN*j*k )
                yt[k] += y[j] * expTerm
    else:                        # Fast Fourier transform
        yt = np.fft.fft(y)

    stopTime = time.time()
    return(stopTime - startTime)

npts = np.logspace(2.5,5.7, base=10.0)
fft_times = np.empty(len(npts),dtype='float64')
for i,n in enumerate(npts):
#    print(i,n)
    fft_times[i] = time_fft(int(n), 0.2, 0, 2)

fig, ax = plt.subplots()
ax.scatter(npts, fft_times)
ax.set_xlabel('Number of points')
ax.set_ylabel('Time to compute [s]')
plt.show()