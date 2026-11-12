# %% [markdown]
"""

#Tarea Semanal 4

Haremos el siguiente experimento:

    Senoidal de frecuencia $$f_0 = k_0 \cdot \Delta_f$$
    potencia normalizada, es decir energía (o varianza) unitaria

Se pide:

a) Sea $$k_0 = \left\{ \frac{N}{4}, \frac{N}{4}+0.25, \frac{N}{4}+0.5   \right\}$$. Notar que a cada senoidal se le agrega una pequeña desintonía respecto a $$\Delta_f$$ . Graficar las tres densidades espectrales de potencia (PDS's) y discutir cuál es el efecto de dicha desintonía en el espectro visualizado.
b) Verificar la potencia unitaria de cada PSD. En base a la teoría estudiada, ¿cuál es la razón por la cual una señal senoidal tiene un espectro (PSD) tan diferente respecto a otra de muy pocos Hertz de diferencia?

Bonus:
Repetir el experimento mediante la técnica de zero padding. Dicha técnica consiste en agregar ceros al final de la señal para aumentar de forma ficticia. Probar agregando un vector de 9*N ceros al final. Discuta los resultados obtenidos.
"""

# %% [markdown]
"""
## Inicio del codigo
"""

# %% Import and directives

import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
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

#%% [markdown]
"""
### miSinMat

Genera una matriz donde cada fila es una senoidal correspondiente a una frecuenca del vector ff 
"""
#%%

def miSinMat(vmax, dc, ff, ph, nn, fs, snr):
    tt = np.arange(nn) * 1/fs
    xx = dc + vmax * np.sin( 2 * np.pi * np.outer(ff,tt) + ph)

    if (snr != -1):
        Pna = (vmax**2/2) / undB(snr)    
        xx = np.random.normal(0, np.sqrt(Pna), len(ff), nn) 
  
    return (tt, xx)

# %% [markdown]
"""

## Generacion de la matriz

Se toma un numero arbitrario de puntos en [N/4; N/4 + df) para la Matriz de Senos

Se computan las FFts

Producto entre un Vector de Unos y una matriz es la integral columna a columna de

$$ B = \left(\mathbf{1}_m^{\top} A\right) $$

$$ B_j = \left(\mathbf{1}_m^{\top} A\right)_j = \sum_{i=1}^{m} (\mathbf{1}_m)_i \, a_{ij} = \sum_{i=1}^{m} 1 \cdot a_{ij} = \sum_{i=1}^{m} a_{ij} = \left ( \sum_{i=1}^{m} a_{i} \right ){}_j $$  

Tambien se computan los mismos Valores con un Zero Padding de 10N

"""

# %% 

kk = N/4 + np.arange(40)/40
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

NZP = 10*N
XXzp = np.fft.fft(xx, n=NZP)/NZP
FFzp = np.arange(NZP//2)/NZP * fs
WWzp = np.ones(NZP//2) @ abs(XXzp.transpose()[:NZP//2,:]**2) * 2 *10 #*10 para compensar que estamos recortando 10 veces mas el espectro

# %% [markdown]
"""

## Ploteos

### Plot 1: Espectro de cada Señal

Ploteando el espectro de cada señal podemos ver como todas varian en su amplitud en el punto maximo, y en como aumenta el piso de ruido debido al leakeage espectral

"""
# %%
plt.clf()
fig = plt.figure(figsize=(15, 8))
gs = grd.GridSpec(2, 2, figure=fig, width_ratios=[2, 1])

ax     = fig.add_subplot(gs[:, 0])
axZoom = fig.add_subplot(gs[0, 1])
axLeg  = fig.add_subplot(gs[1, 1])
axLeg.axis('off')          # panel used only to host the legend

for i in range(len(XX[:,0])):
  ax.plot(FF, 20*np.log10(2*abs(XX.transpose()[:N//2,i])), ':.', label=f'{ff[i]:.3f} Hz')

for i in range(len(XX[:,0])):
  axZoom.plot(FF, 20*np.log10(2*abs(XX.transpose()[:N//2,i])), ':.', label=f'{ff[i]:.3f} Hz')

ax.set_title(f'FFTs de las senoidales de frecuencia ff')
ax.set_xlabel('Freq (S)')
ax.set_ylabel('Amplitud (W)')

axZoom.set_title(f'Zoom')
axZoom.set_xlabel('Freq (S)')
axZoom.set_ylabel('Amplitud (W)')

ax.grid(axis='both', linestyle='-', alpha=0.4)
axZoom.grid(axis='both', linestyle='-', alpha=0.4)

ax.set_xlim(N/4 - 5, N/4 + 5)
axZoom.set_xlim(N/4 - 2*dt, N/4 + 3*dt)
axZoom.set_ylim(-10, 10)

ax.vlines(x=N/4 - 2*dt, ymin=-10, ymax=10, linestyles="--", colors='red')
ax.vlines(x=N/4 + 3*dt, ymin=-10, ymax=10, linestyles="--", colors='red')
ax.hlines(y=-10, xmin=N/4 - 2*dt, xmax=N/4 + 3*dt, linestyles="--", colors='red')
ax.hlines(y=10, xmin=N/4 - 2*dt, xmax=N/4 + 3*dt, linestyles="--", colors='red')

handles, labels = ax.get_legend_handles_labels()
axLeg.legend(handles, labels, loc='center', title='ff', ncol=2, fontsize=8)

plt.tight_layout()
plt.show()


# %% [markdown]
"""

### Plot 2: Espectro de cada Señal con Zero Padding

Repitiendo el mismo Ploteo podemos notar como al incorporar Zero Padding las distintas señales se ven identicas, variando solo la frecuencia donde se encuentra su maximo.
Para propositos de estimacion espectral es mucho mas util.

Adicionalmente podemos notar la interpolacion del kernel de dirichlet
"""

# %%

plt.clf()
fig = plt.figure(figsize=(15, 8))
gs = grd.GridSpec(2, 2, figure=fig, width_ratios=[2, 1])

ax     = fig.add_subplot(gs[:, 0])
axZoom = fig.add_subplot(gs[0, 1])
axLeg  = fig.add_subplot(gs[1, 1])
axLeg.axis('off')          # panel used only to host the legend

for i in range(len(XXzp[:,0])):
  ax.plot(FFzp, 20*np.log10(2*abs(XXzp.transpose()[:NZP//2,i])), ':.', label=f'{ff[i]:.3f} Hz')

for i in range(len(XXzp[:,0])):
  axZoom.plot(FFzp, 20*np.log10(2*abs(XXzp.transpose()[:NZP//2,i])), ':.', label=f'{ff[i]:.3f} Hz')

ax.set_title(f'FFTs de las senoidales de frecuencia ff')
ax.set_xlabel('Freq (S)')
ax.set_ylabel('Amplitud (W)')

axZoom.set_title(f'Zoom')
axZoom.set_xlabel('Freq (S)')
axZoom.set_ylabel('Amplitud (W)')

ax.grid(axis='both', linestyle='-', alpha=0.4)
axZoom.grid(axis='both', linestyle='-', alpha=0.4)

ax.set_xlim(N/4 - 5, N/4 + 5)
axZoom.set_xlim(N/4 - 2*dt, N/4 + 3*dt)
axZoom.set_ylim(-80, 0)

ax.vlines(x=N/4 - 2*dt, ymin=-80, ymax=0, linestyles="--", colors='red')
ax.vlines(x=N/4 + 3*dt, ymin=-80, ymax=0, linestyles="--", colors='red')
ax.hlines(y=-80, xmin=N/4 - 2*dt, xmax=N/4 + 3*dt, linestyles="--", colors='red')
ax.hlines(y=0, xmin=N/4 - 2*dt, xmax=N/4 + 3*dt, linestyles="--", colors='red')

handles, labels = ax.get_legend_handles_labels()
axLeg.legend(handles, labels, loc='center', title='ff', ncol=2, fontsize=8)

plt.tight_layout()
plt.show()


"""

### Plot 3: Potencia Total

Ploteando la $$f_0$$ contra la integral en todo el espectro de sus amoplitudes podemos que ver que la variacion es en el pero de los casos del orden del 0,1% de la esperada. Esto es asi independientemente del uso de Zero Padding

"""
# %%

plt.clf()
plt.figure(figsize=(10, 8))

ax = plt.subplot(1,1,1)

ax.plot(ff, WW, ':.', color='red', label='No Zero Padding')
ax.plot(ff, WWzp, ':.', color='blue', label='Zero Padding')

ax.set_title(f'Energia Total; Con y Sin Zero Padding')
ax.set_xlabel('Freq (S)')
ax.set_ylabel('Amplitud (W)')

ax.grid(axis='both', linestyle='-', alpha=0.4)

ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# %%