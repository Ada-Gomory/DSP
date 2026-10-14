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
    ffr = (np.pi/2 + (2*np.pi/nn)*np.random.uniform(-2.3, 2.3, rr))
    xx = dc + vmax * np.sin( 2 * np.pi * np.outer(ffr, tt) + ph)
    if (snr != -1):
        Pna = vmax**2/2 / undB(snr)
        xx = xx + np.random.normal(0, np.sqrt(Pna), (nn, rr)) 
    return (tt, xx)

#%% [markdown]
"""
## Test de la funcion

Se utiliza la funcion de fft provista por numpy en azul punteado para corroborar que el resultado es identico al esperado
"""

# %% 
rr = 200

xx = np.zeros((200, 1000, 4))
tt, xx[:,:,0] = miSinMat(
    vmax = np.sqrt(2),
    nn = N,
    rr = rr,
    snr = -1
)
xx[:,:,1] = xx[:,:,0] * np.outer(np.ones(rr), sig.windows.blackmanharris(N, sym='false'))
xx[:,:,2] = xx[:,:,0] * np.outer(np.ones(rr), sig.windows.flattop(N, sym='false'))
xx[:,:,3] = xx[:,:,0] * np.outer(np.ones(rr), sig.windows.barthann(N, sym='false'))
labels = ['Rectangular', 'Blackman-Harris', 'Flattop', 'Barthann']


XX = np.fft.fft(xx, axis = 1, n = 1000)/N
FF = np.arange(N/2)# * 2*np.pi/N

# %%

plt.clf()
plt.figure(figsize=(10, 8))

pltrec = plt.subplot(2, 2, 1, )
pltbla = plt.subplot(2, 2, 2, sharex=pltrec, sharey=pltrec)
pltfla = plt.subplot(2, 2, 3, sharex=pltrec, sharey=pltrec)
pltbar = plt.subplot(2, 2, 4, sharex=pltrec, sharey=pltrec)

for i, ax in enumerate((pltrec, pltbla, pltfla, pltbar)):

  ax.plot(FF[2*N//10:3*N//10]-FF[N//4], 20*np.log10(2*abs(XX.transpose()[i,2*N//10:3*N//10,:])), lw=0.5, alpha=0.6)
  ax.axhline(y = 10*np.log10(2), xmin=0, xmax=1, linestyle=':', color='red', lw=1)

  ax.set_title(f'{labels[i]}')
  ax.set_xlabel('f - $f_0$ [HZ]')
  ax.set_ylabel('Potecia [W/Hz]')

  ax.grid(alpha=0.5)

plt.tight_layout()
plt.show()

# %%

sAmp = abs(XX[:,N//4,:])
sFreq = np.argmax(XX[:,:N//2,:], 1)

# %%

binsa = np.linspace(sAmp.min(), sAmp.max(), 40)
binsf = np.arange(sFreq.min() - 0.5, sFreq.max() + 1.5, 1)

plt.figure(figsize=(12, 5))

axA = plt.subplot(1, 2, 1)
axF = plt.subplot(1, 2, 2)

for i, lab in enumerate(labels):
    axA.hist(sAmp[:, i], bins=binsa, alpha=0.5, label=lab)

axA.set_xlabel('|X[N/4]|')
axA.set_ylabel('Cantidad de realizaciones')
axA.set_title('Estimador de amplitud')
axA.legend()
axA.grid(alpha=0.5)

# sFreq is integer-valued, so put bin edges halfway between integers

for i, lab in enumerate(labels):
    axF.hist(sFreq[:, i], bins=binsf, alpha=0.5, label=lab)

axF.set_xlabel('Bin de la Maima Amplitud')
axF.set_ylabel('Cantidad de realizaciones')
axF.set_title('Estimador de frecuencia')
axF.legend()
axF.grid(alpha=0.5)

plt.tight_layout()
plt.show()

# %%
sesgo_A = np.average(sAmp, 0) - np.sqrt(2)
sesgo_F = np.average(sFreq, 0) - N/4

# %%


plt.figure(figsize=(12, 5))

axA = plt.subplot(1, 2, 1)
axF = plt.subplot(1, 2, 2)

for i, lab in enumerate(labels):
    axA.plot(abs(sesgo_A[i]), np.var(sAmp[:,i]), '.', alpha=1, lw=4, label=lab)

axA.set_xlabel('Sesgo')
axA.set_ylabel('Varianza')
axA.set_title('Sesgo y Varianza de Energia')
axA.legend()
axA.grid(alpha=0.5)

for i, lab in enumerate(labels):
    axF.plot(abs(sesgo_F[i]), np.var(sFreq[:,i]), '.', alpha=1, lw=i, label=lab)

axF.set_xlabel('Sesgo')
axF.set_ylabel('Varianza')
axF.set_title('Sesgo y Varianza de Frecuencia')
axF.legend()
axF.grid(alpha=0.5)

plt.tight_layout()
plt.show()


#%% [markdown]
"""

"""
