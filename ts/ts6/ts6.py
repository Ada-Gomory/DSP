# %% [markdown]
"""
En el repositorio PDStestbench encontrará tres tipos de señales registradas:

- **Electrocardiograma (ECG).** En el archivo **ECG_TP4.mat** encontrará un registro electrocardiográfico (ECG) registrado durante una prueba de esfuerzo, junto con una serie de variables descriptas más abajo.
- **Pletismografía (PPG).** El archivo PPG.csv contiene una señal registrada en reposo de un estudiante de la materia que ha donado su registro para esta actividad.
- **Audio**. Tres registros en los que el profesor pronuncia una frase, y otros dos en los que se silba una melodía muy conocida.

Los detalles de cómo acceder a dichos registros los pueden encontrar en lectura_sigs.py

Se pide:

**1)** Realizar la estimación de la densidad espectral de potencia (PSD) de cada señal mediante alguno de los métodos vistos en clase (Periodograma ventaneado, Welch, Blackman-Tukey).

**2)** Realice una estimación del ancho de banda de cada señal y presente los resultados en un tabla para facilitar la comparación.
Bonus:

-  Proponga algún tipo de señal que no haya sido analizada y repita el análisis. No olvide explicar su origen y cómo fue digitalizada.

"""

# %% Import and directives

import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

import scipy.signal as sig
import scipy.stats as ssta
import scipy.io as sio
from scipy.io.wavfile import write

#!%matplotlib qt

# %% [markdown]
"""
## Inicio del codigo
"""

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

#%% [markdown]
"""
### Lectura de pletismografia

Tomo la lectura de pdstestbench

"""

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

ks  = 2**(np.arange(6) + 1)
maxLen  = (len(xx) // np.min(ks)) // 2 + 1        # longest FF/PXX
thresh = 0.98

FF  = np.full((len(ks), maxLen), np.nan)
PXX = np.full((len(ks), maxLen), np.nan)

#FFlow = np.full(len(ks), 0)
FFHigh = np.full(len(ks), 0)


for i, k in enumerate(ks):
  nps = len(xx) // k
  FF[i, :(nps//2)+1], PXX[i, :(nps//2)+1] = sig.welch(xx, fs_ppg, nperseg=nps, noverlap=nps//2)

PXXthresh = np.sum(PXX, 1) * thresh

for i, k in enumerate(ks):
  acc = 0
  for h in range(len(PXX[i, :][~np.isnan(PXX[i, :])])):
    acc += PXX[i, h]
    if (acc > PXXthresh[i]):
      FFHigh[i] = FF[i, h]
      break


fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1)

prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [prop_colors[i % len(prop_colors)] for i in range(len(ks))]

for i, k in enumerate(ks):
  ax.plot(FF[i,:], abs(PXX[i,:]), '-', lw=0.7, alpha=0.4, label=f'{k} ventanas', color=colors[i])

  ax.add_patch(ptch.Rectangle((0, 0), FFHigh[i], 1000,
               facecolor=colors[i], alpha=0.05, hatch='x',
               edgecolor=colors[i], lw=1.5))

plt.tight_layout()
plt.legend()
plt.show()

# %%
