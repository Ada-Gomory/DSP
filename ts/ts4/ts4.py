# %% [markdown]
"""

#Tarea Semanal 3

> En esta tarea simularemos el comportamiento de un ADC y cómo sus parámetros, tamaño de palabra de B bits y frecuencia de muestreo fs afectan a una señal (aproximadamente) analógica.
>
> Para ello se puede analizar una senoidal con los siguientes parámetros:
> - Frecuencia $f_0$ arbitraria, por ejemplo $f_0 = f_s/N = \Delta f$, 
> - Potencia normalizada, es decir unitaria
>
> Se pide diseñar un bloque cuantizador que opere sobre una señal discreta en tiempo , de forma tal que para un ADC de B bits y rango $V_{FS} = 2 \cdot V_F$, el operador 
> $$s_Q = \underset{B,V_{FS}}{Q} s_R$$
> generará una  comprendida entre $\pm V_F$ y con valores discretos establecidos por el paso de cuantización $q = \frac{V_{FS}}{2^B}$(Volts).
>
> Visualice en una misma gráfica  y , donde se pueda observar que tienen el mismo rango en Volts y el efecto de la cuantización para Volts y  B = 4, 8 y 16 bits.
>
> Bonus
> - Analizar la señal de error $e = s_Q - s_R$ verificando las descripciones estadísticas vistas en teoría (Distribución uniforme, media, varianza, incorrelación)
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

Inicialmente defino funciones de uso general e incorporo codigo de previas TS que sera util 
"""

# %% General Defs 

def undB (dB):
    return 10**(dB/10)

def dB (x):
    return 10*np.log10(x)

j = complex(0, 1)

#Params Generales 
fs = 2**10      #500Hz de BW
ts = 1/fs
N = fs        #DeltaF = 1Hz; norm
df = fs/N
dt = 1/df

#
vfs = 1.65
B = 6

#Params func 
vmax = 0.85
dc = 0.          
ff = 5
ph = 0.
snr = 30

def miSinMat (vmax, dc, ff, ph, nn, fs, snr):
    tt = np.arange(nn) * 1/fs
    xx = dc + vmax * np.sin( 2 * np.pi * np.outer(ff,tt) + ph)

    if (snr != -1):
        Pna = (vmax**2/2) / undB(snr)    
        xx = np.random.normal(0, np.sqrt(Pna), len(ff), nn) 
  
    return (tt, xx)

# %% [markdown]
"""
"""

# %% 

kk = N/4 + np.arange(10)/10
ff = kk * df

tt, xx = miSinMat(
    vmax = np.sqrt(2),
    dc = 0,
    ff = ff,
    ph = 0,
    nn = N,
    fs = fs,
    snr = -1)

XX = np.fft.fft(xx)/N
FF = np.arange(N//2)/N * fs
WW = np.ones(N//2) @ abs(XX.transpose()[:N//2,:]**2) *2

# %%

plt.clf()
plt.figure(figsize=(10, 8))

ax = plt.subplot(1,1,1)

for i in range(len(XX[:,0])):
  ax.plot(FF, 20*np.log10(2*abs(XX.transpose()[:N//2,i])), ':.', label=ff[i])

ax.set_title(f'FFTs de las senoidales de frecuencia ff')
ax.set_xlabel('Freq (S)')
ax.set_ylabel('Amplitud (W)')

ax.grid(axis='both', linestyle='-', alpha=0.4)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# %%

plt.clf()
plt.figure(figsize=(10, 8))

ax = plt.subplot(1,1,1)

ax.plot(ff, WW, ':.')

ax.set_title(f'Energia Total')
ax.set_xlabel('$f_f$ (S)')
ax.set_ylabel('$\int_0^{f_s} f_f $')

ax.grid(axis='both', linestyle='-', alpha=0.4)

plt.tight_layout()
plt.show()

# %%
NZP = 10*N
XX = np.fft.fft(xx, n=NZP)/NZP
FF = np.arange(NZP//2)/NZP * fs
WW = np.ones(NZP//2) @ abs(XX.transpose()[:NZP//2,:]**2) *2

# %%

plt.clf()
plt.figure(figsize=(10, 8))

ax = plt.subplot(1,1,1)

for i in range(len(XX[:,0])):
  ax.plot(FF, 20*np.log10(2*abs(XX.transpose()[:NZP//2,i])), ':.', label=ff[i])

ax.set_title(f'FFTs de las senoidales de frecuencia ff; Zero padding aplicado ')
ax.set_xlabel('Freq (S)')
ax.set_ylabel('Amplitud (W)')

ax.grid(axis='both', linestyle='-', alpha=0.4)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# %%

plt.clf()
plt.figure(figsize=(10, 8))

ax = plt.subplot(1,1,1)

ax.plot(ff, WW, ':.')

ax.set_title(f'Energia Total; Zero Padding')
ax.set_xlabel('Freq (S)')
ax.set_ylabel('Amplitud (W)')

ax.grid(axis='both', linestyle='-', alpha=0.4)

plt.tight_layout()
plt.show()

# %%