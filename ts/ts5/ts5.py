# %% [markdown]
"""
# Tarea Semanal 5

En esta tarea continuaremos con el capítulo 14 de Holton. DSP Principles and App.

Comenzaremos con la generación de la siguiente señal:

$$
x(n) = a_0 \cdot \mathrm{sen}(\Omega_1 \cdot n) + n_a(n)
$$

siendo

$$
\Omega_1 = \Omega_0 + f_r \cdot \frac{2\pi}{N}
$$

$$
\Omega_0 = \frac{\pi}{2}
$$

siendo la variable aleatoria definida por la siguiente distribución de probabilidad

$$
f_r \sim \mathcal{U}(-2,\ 2)
$$

$$
n_a \sim \mathcal{N}(0,\ \sigma^2)
$$

Diseñe los siguientes estimadores, de amplitud $a_1$

$$
\hat{a}_1^i = \left| X_w^i(\Omega_0) \right| = \left| \mathcal{F}\{x(n) \cdot w_i(n)\} \right|
$$

para la $i$-ésima realización y la $w$-ésima ventana (ver detalles debajo).

y de frecuencia $\Omega_1$

$$
\hat{\Omega}_1^i = \arg\max_{\Omega} \left\{ \left| X_w^i(\Omega) \right| \right\}
$$

para cada una de las ventanas:

para cada una de las ventanas:

- **rectangular** (sin ventana),
- **flattop**
- **blackmanharris**
- **otra** que elija de scipy.signal.windows

Y siguiendo las siguientes consignas para su experimentación:

- Considere 200 realizaciones (muestras tomadas de $f_r$) de 1000 muestras para cada experimento.
- Parametrice para SNR's de 3 y 10 db (**Ayuda:** calibre $a_0$ para que la potencia de la senoidal sea 1 W).

Se pide:

1) Realizar una tabla por cada SNR, que describa el sesgo y la varianza de cada estimador para cada ventana analizada. Recuerde incluir las ventanas **rectangular** (sin ventana), **flattop** y **blackmanharris** y otras que considere.


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
from matplotlib.widgets import CheckButtons
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
## MySinMat

Voy a usar varias funciones arbitrarias para testear la DSFT, asi que tiro aca el codigo de la ts4, modificado para tomar ff aleatoria y numero de realizaciones

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
## Simulacion

Se calcula una matriz de rr x tt, conteniendo en cada columna un seno de frecuencia aleatoria con ruido 

Se lleva la matriz a una dimencional, apilando las multiplicaciones por cada ventana en la tercer dimension

Se calcula la FFt, los estimadores de amplitud y frecuencia para cada ventana y realizacion

Se calcula el sesgo de cada estimador
"""

# %% 
rr = 200
windows = ['rect', 'triangle', 'blackmanharris', 'flattop', 'barthann', 'cosine']
colors =  ['red', 'blue', 'orange', 'green', 'purple', 'teal']

xx = np.zeros((rr, N, len(windows)))
tt, xx[:,:,0] = miSinMat(
    vmax = np.sqrt(2),
    nn = N,
    rr = rr,
    snr = -1
)

for i, win in enumerate(windows):
  xx[:,:,i] = xx[:,:,0] * np.outer(np.ones(rr), sig.windows.get_window(win, N))


XX = np.fft.fft(xx, axis = 1, n = N)/N
FF = np.arange(N/2)# * 2*np.pi/N

sAmp = abs(XX[:,N//4,:]) * np.sqrt(2)
sFreq = np.argmax(np.abs(XX[:,:N//2,:]), 1)

sesgo_A = np.average(sAmp, 0) - 1
sesgo_F = np.average(sFreq, 0) - N/4

#%% [markdown]
"""
## Plots

### ffts

Primero graficamos la fft de cada realizacion individaulmente, separandolas por ventana, para poder ver la forma de las distintas ventanas

Ampliando los picos podemos notar que la rectangular es la unica donde los picos llegan a la linea roja representando el valor que sabemos que deberiamos ver en el pico, pero es tambien la que tiene mayor diferencia entre los picos y los valles. Esto tiene sentido ya que sabemos que es la que tiene mayor varianza y menor sesgo

Al mismo tiempo, podemos ver que la flattop es la mas lejana al valor esperado, pero tiene muy baja varianza entre las muestras

"""
# %%

fig, (plts) = plt.subplots(2, (1+len(windows))//2 ,figsize=(15, 12), sharex=True, sharey=True)

for i, ax in enumerate(plts.flat):
  if (i >= len(windows)):
      ax.axis('off')
      continue
  
  ax.plot(FF, 20*np.log10(2*abs(XX.transpose()[i,:N//2,:])), lw=0.5, alpha=0.6)
  ax.axhline(y = 10*np.log10(2), xmin=0, xmax=1, linestyle=':', color='red', lw=1)

  ax.set_title(f'{windows[i]}')
  ax.set_xlabel('f - $f_0$ [HZ]')
  ax.set_ylabel('Potecia [dBW/Hz]')

  ax.set_xlim(N/4 - 15, N/4 + 15)

  ax.grid(alpha=0.5)

plt.tight_layout()
plt.show()


#%% [markdown]
"""

### Histogramas

Se hacen los histogramas con los estimadores de cada realizacion, y se grafica la media y varianza de cada uno. Podemos corroborar lo que habiamos observado previamente, en terminos de varianza la flattop resulta la mejor y la rectangular la peor

"""
# %%

binsa = np.linspace(sAmp.min(), sAmp.max(), 40)
binsf = np.arange(sFreq.min() - 0.5, sFreq.max() + 1.5, 1)

prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(15, 5))
gs = grd.GridSpec(1, 3, figure=fig, width_ratios=[3, 3, 1])

axA   = fig.add_subplot(gs[0, 0])
axF   = fig.add_subplot(gs[0, 1])
axLeg = fig.add_subplot(gs[0, 2])
axLeg.axis('off')   # blank panel that hosts the checkboxes

artists = [[] for _ in windows]

def addMeanStd(ax, data, color, row, nrows):
    m, s = np.mean(data), np.var(data)
    y = 0.97 - 0.045*row  
    return ax.plot([m - s, m + s], [y, y], color=color, lw=2,
                   marker='|', ms=8, mew=2,
                   transform=ax.get_xaxis_transform())[0], \
           ax.plot(m, y, 'o', color=color, ms=5,
                   transform=ax.get_xaxis_transform())[0]

for i, lab in enumerate(windows):
    _, _, bars = axA.hist(sAmp[:, i], bins=binsa, alpha=0.5, color=colors[i])
    artists[i] += list(bars)
    artists[i] += addMeanStd(axA, sAmp[:, i], colors[i], i, len(windows))

for i, lab in enumerate(windows):
    _, _, bars = axF.hist(sFreq[:, i], bins=binsf, alpha=0.5, color=colors[i])
    artists[i] += list(bars)
    artists[i] += addMeanStd(axF, sFreq[:, i], colors[i], i, len(windows))

axA.set_xlabel('|X[N/4]|')
axA.set_ylabel('Cantidad de realizaciones')
axA.set_title('Estimador de amplitud')
axA.grid(alpha=0.5)

axF.set_xlabel('Bin de la Maxima Amplitud')
axF.set_ylabel('Cantidad de realizaciones')
axF.set_title('Estimador de frecuencia')
axF.grid(alpha=0.5)

#################################
## Checkboxes
#################################
axLeg.set_title('Ventana')
chk = CheckButtons(axLeg, windows, [True] * len(windows),
                   check_props={'facecolor': colors},
                   frame_props={'edgecolor': colors})
for txt, c in zip(chk.labels, colors):
    txt.set_color(c)

def toggle(label):
    i = windows.index(label)
    for art in artists[i]:
        art.set_visible(not art.get_visible())
    fig.canvas.draw_idle()

chk.on_clicked(toggle)
fig._chk = chk   # keep a reference, or the widget is garbage collected and stops responding

plt.tight_layout()
plt.show()


#%% [markdown]
"""

### Scatter

Se comparan los valores de sesgo y varianza obtenidos
"""
# %%


plt.figure(figsize=(12, 5))

axA = plt.subplot(1, 2, 1)
axF = plt.subplot(1, 2, 2)

for i, lab in enumerate(windows):
    axA.plot(abs(sesgo_A[i]), np.var(sAmp[:,i]), 'o', alpha=1, lw=4, label=lab, color=colors[i])

axA.set_xlabel('Sesgo')
axA.set_ylabel('Varianza')
axA.set_title('Sesgo y Varianza de Energia')
axA.grid(alpha=0.5)

for i, lab in enumerate(windows):
    axF.plot(abs(sesgo_F[i]), np.var(sFreq[:,i]), 'o', alpha=1, lw=i, label=lab, color=colors[i])

axF.set_xlabel('Sesgo')
axF.set_ylabel('Varianza')
axF.set_title('Sesgo y Varianza de Frecuencia')
axF.legend()
axF.grid(alpha=0.5)

plt.tight_layout()
plt.show()


#%% [markdown]
"""
## Mejoras al estimador

### Correccion de Sesgo

Para elimir el sesgo de amplitud introducido al multiplicar por la ventana, integramos la ventana, y dividimos el numero de muestras para obtener el sesgo introducido en la multiplicacion, y dividimos xx por este valor
Luego repetimos los pasos anteriores

"""
# %% 

xxw = np.zeros((rr, N, len(windows)))

for i, win in enumerate(windows):
  xxw[:,:,i] = xx[:,:,i] / ((np.ones(N) @ sig.windows.get_window(win, N))/N)

XXW = np.fft.fft(xxw, axis = 1, n = N)/N
FFW = np.arange(N/2)# * 2*np.pi/N

sAmpW = abs(XXW[:,N//4,:]) * np.sqrt(2)
sFreqW = np.argmax(np.abs(XXW[:,:N//2,:]), 1)

sesgo_AW = np.average(sAmpW, 0) - 1
sesgo_FW = np.average(sFreqW, 0) - N/4

# %%

fig, (plts) = plt.subplots(2, (1+len(windows))//2 ,figsize=(15, 12), sharex=True, sharey=True)

for i, ax in enumerate(plts.flat):
  if (i >= len(windows)):
      ax.axis('off')
      continue
  
  ax.plot(FFW, 20*np.log10(2*abs(XXW.transpose()[i,:N//2,:])), lw=0.5, alpha=0.6)
  ax.axhline(y = 10*np.log10(2), xmin=0, xmax=1, linestyle=':', color='red', lw=1)

  ax.set_title(f'{windows[i]}')
  ax.set_xlabel('f - $f_0$ [HZ]')
  ax.set_ylabel('Potecia [dBW/Hz]')

  ax.set_xlim(N/4 - 15, N/4 + 15)

  ax.grid(alpha=0.5)

plt.tight_layout()
plt.show()

# %%

binsa = np.linspace(sAmpW.min(), sAmpW.max(), 40)
binsf = np.arange(sFreqW.min() - 0.5, sFreqW.max() + 1.5, 1)

prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(15, 5))
gs = grd.GridSpec(1, 3, figure=fig, width_ratios=[3, 3, 1])

axA   = fig.add_subplot(gs[0, 0])
axF   = fig.add_subplot(gs[0, 1])
axLeg = fig.add_subplot(gs[0, 2])
axLeg.axis('off')   # blank panel that hosts the checkboxes

artists = [[] for _ in windows]

def addMeanStd(ax, data, color, row, nrows):
    m, s = np.mean(data), np.var(data)
    y = 0.97 - 0.045*row  
    return ax.plot([m - s, m + s], [y, y], color=color, lw=2,
                   marker='|', ms=8, mew=2,
                   transform=ax.get_xaxis_transform())[0], \
           ax.plot(m, y, 'o', color=color, ms=5,
                   transform=ax.get_xaxis_transform())[0]

for i, lab in enumerate(windows):
    _, _, bars = axA.hist(sAmpW[:, i], bins=binsa, alpha=0.5, color=colors[i])
    artists[i] += list(bars)
    artists[i] += addMeanStd(axA, sAmpW[:, i], colors[i], i, len(windows))

for i, lab in enumerate(windows):
    _, _, bars = axF.hist(sFreqW[:, i], bins=binsf, alpha=0.5, color=colors[i])
    artists[i] += list(bars)
    artists[i] += addMeanStd(axF, sFreqW[:, i], colors[i], i, len(windows))

axA.set_xlabel('|X[N/4]|')
axA.set_ylabel('Cantidad de realizaciones')
axA.set_title('Estimador de amplitud')
axA.grid(alpha=0.5)

axF.set_xlabel('Bin de la Maxima Amplitud')
axF.set_ylabel('Cantidad de realizaciones')
axF.set_title('Estimador de frecuencia')
axF.grid(alpha=0.5)

#################################
## Checkboxes
#################################
axLeg.set_title('Ventana')
chk = CheckButtons(axLeg, windows, [True] * len(windows),
                   check_props={'facecolor': colors},
                   frame_props={'edgecolor': colors})
for txt, c in zip(chk.labels, colors):
    txt.set_color(c)

def toggle(label):
    i = windows.index(label)
    for art in artists[i]:
        art.set_visible(not art.get_visible())
    fig.canvas.draw_idle()

chk.on_clicked(toggle)
fig._chk = chk   # keep a reference, or the widget is garbage collected and stops responding

plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(12, 5))

axA = plt.subplot(1, 2, 1)
axF = plt.subplot(1, 2, 2)

for i, lab in enumerate(windows):
    axA.plot(abs(sesgo_AW[i]), np.var(sAmpW[:,i]), 'o', alpha=1, lw=4, label=lab, color=colors[i])

axA.set_xlabel('Sesgo')
axA.set_ylabel('Varianza')
axA.set_title('Sesgo y Varianza de Energia')
axA.grid(alpha=0.5)

for i, lab in enumerate(windows):
    axF.plot(abs(sesgo_FW[i]), np.var(sFreqW[:,i]), 'o', alpha=1, lw=i, label=lab, color=colors[i])

axF.set_xlabel('Sesgo')
axF.set_ylabel('Varianza')
axF.set_title('Sesgo y Varianza de Frecuencia')
axF.legend()
axF.grid(alpha=0.5)

plt.tight_layout()
plt.show()


#%% [markdown]
"""
En el primer plot podemos ver que esto corrige exactamente el error introducido, haciendo que los maximos alcancen el nivel esperado. En los plots 2 y 3 podemos notar que que esto no elimina completamente el sesgo, solamente lo reduce. El resto del sesgo se debe al leakage espectral

### Zero Padding

Finalmente agregamos zero padding y repetimos los pasos anteriores

"""


# %% 
NZ = 10*N

XXZ = np.fft.fft(xx, axis = 1, n = NZ)/N
FFZ = np.arange(NZ/2)/10# * 2*np.pi/N

sAmpZ = abs(XXZ[:,NZ//4,:]) * np.sqrt(2)
sFreqZ = np.argmax(np.abs(XXZ[:,:NZ//2,:]), 1)/10

sesgo_AZ = np.average(sAmpZ, 0) - 1
sesgo_FZ = np.average(sFreqZ, 0) - N/4

# %%

fig, (plts) = plt.subplots(2, (1+len(windows))//2 ,figsize=(15, 12), sharex=True, sharey=True)

for i, ax in enumerate(plts.flat):
  if (i >= len(windows)):
      ax.axis('off')
      continue
  
  ax.plot(FFZ, 20*np.log10(2*abs(XXZ.transpose()[i,:NZ//2,:])), lw=0.5, alpha=0.6)
  ax.axhline(y = 10*np.log10(2), xmin=0, xmax=1, linestyle=':', color='red', lw=1)

  ax.set_title(f'{windows[i]}')
  ax.set_xlabel('f - $f_0$ [HZ]')
  ax.set_ylabel('Potecia [dBW/Hz]')

  ax.set_xlim(N/4 - 15, N/4 + 15)

  ax.grid(alpha=0.5)


plt.tight_layout()
plt.show()

# %%

binsa = np.linspace(sAmpZ.min(), sAmpZ.max(), 40)
binsf = np.arange(sFreqZ.min() - 0.5, sFreqZ.max() + 1.5, 1)

prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(15, 5))
gs = grd.GridSpec(1, 3, figure=fig, width_ratios=[3, 3, 1])

axA   = fig.add_subplot(gs[0, 0])
axF   = fig.add_subplot(gs[0, 1])
axLeg = fig.add_subplot(gs[0, 2])
axLeg.axis('off')   # blank panel that hosts the checkboxes

artists = [[] for _ in windows]

def addMeanStd(ax, data, color, row, nrows):
    m, s = np.mean(data), np.var(data)
    y = 0.97 - 0.045*row  
    return ax.plot([m - s, m + s], [y, y], color=color, lw=2,
                   marker='|', ms=8, mew=2,
                   transform=ax.get_xaxis_transform())[0], \
           ax.plot(m, y, 'o', color=color, ms=5,
                   transform=ax.get_xaxis_transform())[0]

for i, lab in enumerate(windows):
    _, _, bars = axA.hist(sAmpZ[:, i], bins=binsa, alpha=0.5, color=colors[i])
    artists[i] += list(bars)
    artists[i] += addMeanStd(axA, sAmpZ[:, i], colors[i], i, len(windows))

for i, lab in enumerate(windows):
    _, _, bars = axF.hist(sFreqZ[:, i], bins=binsf, alpha=0.5, color=colors[i])
    artists[i] += list(bars)
    artists[i] += addMeanStd(axF, sFreqZ[:, i], colors[i], i, len(windows))

axA.set_xlabel('|X[N/4]|')
axA.set_ylabel('Cantidad de realizaciones')
axA.set_title('Estimador de amplitud')
axA.grid(alpha=0.5)

axF.set_xlabel('Bin de la Maxima Amplitud')
axF.set_ylabel('Cantidad de realizaciones')
axF.set_title('Estimador de frecuencia')
axF.grid(alpha=0.5)

#################################
## Checkboxes
#################################
axLeg.set_title('Ventana')
chk = CheckButtons(axLeg, windows, [True] * len(windows),
                   check_props={'facecolor': colors},
                   frame_props={'edgecolor': colors})
for txt, c in zip(chk.labels, colors):
    txt.set_color(c)

def toggle(label):
    i = windows.index(label)
    for art in artists[i]:
        art.set_visible(not art.get_visible())
    fig.canvas.draw_idle()

chk.on_clicked(toggle)
fig._chk = chk   # keep a reference, or the widget is garbage collected and stops responding

plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(12, 5))

axA = plt.subplot(1, 2, 1)
axF = plt.subplot(1, 2, 2)

for i, lab in enumerate(windows):
    axA.plot(abs(sesgo_AZ[i]), np.var(sAmpZ[:,i]), 'o', alpha=1, lw=4, label=lab, color=colors[i])

axA.set_xlabel('Sesgo')
axA.set_ylabel('Varianza')
axA.set_title('Sesgo y Varianza de Energia')
axA.grid(alpha=0.5)

for i, lab in enumerate(windows):
    axF.plot(abs(sesgo_FZ[i]), np.var(sFreqZ[:,i]), 'o', alpha=1, lw=i, label=lab, color=colors[i])

axF.set_xlabel('Sesgo')
axF.set_ylabel('Varianza')
axF.set_title('Sesgo y Varianza de Frecuencia')
axF.legend()
axF.grid(alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
En este caso, no hay cambios muy notables en los estimadores.
"""
