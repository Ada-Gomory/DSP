# %% [markdown]
"""
# Tarea Semanal 1

> En este primer trabajo comenzaremos por diseñar un generador de señales que utilizaremos en las primeras simulaciones que hagamos. La primer tarea consistirá en programar una función que genere señales senoidales y que permita parametrizar:
>
> - la amplitud máxima de la senoidal (volts)
> - su valor medio (volts)
> - la frecuencia (Hz)
> - la fase (radianes)
> - la cantidad de muestras digitalizada por el ADC (# muestras)
> - la frecuencia de muestreo del ADC.
>
> Es decir que la función que uds armen debería admitir se llamada de la siguiente manera <br>
>```tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)```

"""

# %% [markdown]
"""
## Inicio del codigo
"""

# %% Import and directives

import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as sta
#!%matplotlib qt

#%% [markdown]
"""
### Primitivas generales

Inicialmente defino funciones de uso general para mantener el codigo mas adelante limpio

"""

# %% General Defs 

def undB (dB):
    return 10**(dB/10)

def dB (x):
    return 10*np.log10(x)

#%% [markdown]
"""
### Definicion de parametros

En este bloque se definen los parametros de sampleo y los parametros aplicados por defecto en el generador de señales

"""

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


#%% [markdown]
"""
## MySigGen

Voy a usar varias funciones arbitrarias para testear la DSFT, asi que tiro aca el codigo de la ts1

"""

# %% Func Defs 

def miSinMat (vmax, nn, rr, snr):
    tt = np.arange(nn) * 1/fs
    ffr = (np.pi/2 + 2*np.pi*np.random.uniform(-2, 2, rr)/nn)
    xx = dc + vmax * np.sin( 2 * np.pi * np.outer(ffr, tt) + ph)
    if (snr != -1):
        Pna = vmax**2/2 / undB(snr)
        xx = xx + miNoise(Pot = Pna, nn = nn, rr = rr)
    return (tt, xx)

def miNoise (Pot, nn, rr):
    xna = np.random.normal(0, np.sqrt(Pot), (nn, rr)) 
    return xna

#%% [markdown]
"""
## Test de la funcion

Se utiliza la funcion de fft provista por numpy en azul punteado para corroborar que el resultado es identico al esperado
"""

# %% 

tt, xx = miSinMat(
    vmax = np.sqrt(2),
    nn = N,
    rr = 200,
    snr = -1
)

XX = np.fft.fft(xx, n = 1000)/N
FF = np.arange(N/2) * 2*np.pi/N

plt.clf()
plt.figure(figsize=(10, 8))

plt.plot(FF, 20*np.log10(2*abs(XX.transpose()[:N//2,:])), lw=0.5, alpha=0.6)
plt.hlines(y = 10*np.log10(2), xmin=0, xmax=np.pi, linestyles=':', color='red', lw=1)

plt.title(f'...')
plt.xlabel('Frecuencia [HZ]')
plt.ylabel('Potecia [W/Hz]')
plt.grid(linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()


#%% [markdown]
"""

"""
