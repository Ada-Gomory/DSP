# %% [markdown]


# %% Import and directives

import numpy as np
import cmath as cm
import matplotlib.pyplot as plt

import scipy.signal as sig
import scipy.stats as ssta
import scipy.io as sio
from scipy.io.wavfile import write

#!%matplotlib qt

# %% General Defs 

def undB (dB):
    return 10**(dB/10)

def dB (x):
    return 10*np.log10(x)

# %% 

j = complex(0, 1)

#Params Generales 
fs = 2*np.pi      #500Hz de BW
ts = 1/fs
N = 1000        #DeltaF = 1Hz; norm
df = fs/N
dt = 1/df

#
vfs = 1.65
B = 6

#Params func 
vmax = 1.4
dc = 0.          
ff = 5
ph = 0.
snr = 30

# %% Func Defs 



####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# # Cargar el archivo CSV como un array de NumPy
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe




# %% 
N
xx = ppg

ks = 2**np.arange(start=2, stop=6, step=1)

plt.clf()
plt.figure(figsize=(10, 8))

for i, k in enumerate(ks):
  nps=len(xx)//k
  FF, PXX = sig.welch(xx, fs_ppg, nperseg=nps, noverlap = nps//2)
  plt.plot(FF, 20*np.log10(abs(PXX)), lw=0.7, alpha=0.6, label=f'{k}')


plt.tight_layout()
plt.legend()
plt.show()

# %%
