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

# %% Import 

import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import scipy.signal as sig

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
fs = 1024       #500Hz de BW
ts = 1/fs
N = fs        #DeltaF = 1Hz; norm
df = fs/N
dt = 1/df

#Params func 
vmax = 1.        
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

def miSin (vmax, dc, ff, ph, nn, fs):
    tt = np.arange(nn) * 1/fs
    xx = dc + vmax * np.sin( 2 * np.pi * ff * tt + ph)
    pa = vmax**2/2
    return (tt, xx, pa)

def miPWM (vmax, dc, ff, ph, nn, fs, duty):
    tt = np.arange(nn) * 1/fs
    xx = np.arange(nn)
    pp = (tt*ff + ph/(2*np.pi)) % 1 #time as proportion of cycle

    for i in range(len(pp)):
        if(pp[i] < duty):
            xx[i] = vmax
        else:
            xx[i] = 0
    xx = xx + dc - (vmax * duty)
    pa = vmax**2*duty
    return (tt, xx, pa)

def miScalene (vmax, dc, ff, ph, nn, fs, duty):
    tt = np.arange(nn) * 1/fs
    xx = np.arange(nn + 0.0)
    pp = (tt*ff + ph/(2*np.pi)) % 1 #time as proportion of cycle

    for i in range(len(pp)):
        if(pp[i] < duty):
            xx[i] = (vmax/duty) * pp[i] 
        else:
            xx[i] = (vmax/(1-duty)) * (1-pp[i])
    xx = xx + dc - (vmax/2)
    pa = vmax**2/3
    return (tt, xx, pa)

def miNoise (Pot, nn):
    xna = np.random.normal(loc = 0, scale = np.sqrt(Pot), size = nn) 
    return xna

def miSignalGenerator (sigType = "sqr", vmax = vmax, dc = dc, ff = ff, ph = ph, nn = N, fs = fs, snr = -1, duty = 0.5):
    if ((duty > 1) or (duty < 0)):
      print("dutycyle invalido")
      return
    
    if ((snr < 0) and (snr != -1)):
      print("snr invalido")
      return

    match sigType:
        case "sine":
            tt, xx, pa = miSin(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs)

        case "saw":
            tt, xx, pa = miScalene(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs, duty = 1)
        case "revSaw":
            tt, xx, pa = miScalene(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs, duty = 0)
        case "tri":
            tt, xx, pa = miScalene(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs, duty = 0.5)
        case "asTri":
            tt, xx, pa = miScalene(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs, duty = duty)

        case "square":
            tt, xx, pa = miPWM(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs, duty = 0.5)
        case "PWM":
            tt, xx, pa = miPWM(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs, duty = duty)

        case _:
            print("sigType invlaido")
            tt = np.arange(N)
            xx = np.arange(N)
    
    if (snr != -1):
        Pna = pa / undB(snr)
        xx = xx + miNoise(Pot = Pna, nn = nn)
    return (tt, xx)

# %% [markdown]
"""
## miDSFT

Para ahorro computacional defino todos los Factores ${W^{kn}_N}$

$$
\left[WW\right]_{k,n} = {W^{kn}_N}
$$

"""

# %%

WW = np.e**(-j * 2 * np.pi / N)
WW = WW**np.outer(np.arange(N), np.arange(N))

# %% [markdown]
"""

Partiendo de la formula de la DFT

$$
X[k] = \sum_{n=0}^{N-1} xx[n] \cdot e^\frac{-j\cdot 2\pi \cdot k \cdot n}N = \sum_{n=0}^{N-1} x[n] \cdot {W^{kn}_N}
$$

la pasamos a codigo literalmente

```py
def miDFT(xx, fs = fs):
  XX = np.arange(nn) + complex(0,0)
  for k in range(nn):
      acc = 0
      for n in range(nn):
          acc = acc + xx[n] * np.e**(-j * 2 * np.pi * k * n / nn)
      XX[k] = acc
  return XX
```

Dada la matriz precalculada $$[WW]$$


```py
def miDFT(xx, fs = fs):
  nn = len(xx)

  XX = np.arange(nn) + complex(0,0)
  for k in range(nn):
      acc = 0
      for n in range(nn):
          acc = acc + xx[n] * WW[k,n]
      XX[k] = acc
  return XX
```

ahora bien, el bloque 

$$
\sum_{n=0}^{N-1} xx[n] \cdot WW[k, n]
$$

no es otra cosa que la definicion del producto interno de de xx por la k-esima fila de la matriz WW, por lo que podemos eliminar uno de los bucles


```py
def miDFT(xx, fs = fs):
  nn = len(xx)

  XX = np.arange(nn) + complex(0,0)
  for k in range(nn):
      XX [k] = xx @ WW[k,:]
  return XX
```

finalmente, dado que cada valor k del vector xx es un producto de un vector, por la fila k-esima de una matriz, podemos realizar todas los pasos del bucle con una multiplicacion matricial

```py
def miDFT(xx, fs = fs):
  XX = xx @ WW
```

para la implementacion utilizada se calcula tambien el vector de frecuencias, utilizado para el eje de abscisas del plot, y se recalcula WW en caso de que el largo sea distinto de N

"""

# %%
def miDSFT(xx, fs = fs):
  nn = len(xx)
  FF = np.arange(nn) * fs/nn

  if (nn != N):
    ww = np.e**(-j * 2 * np.pi / nn)
    ww = WW**np.outer(np.arange(nn), np.arange(nn))
    XX = xx @ ww
  else:
    XX = xx @ WW

  return FF, XX

#%% [markdown]
"""
## Test de la funcion

Se utiliza la funcion de fft provista por numpy en azul punteado para corroborar que el resultado es identico al esperado
"""

# %% 

tt, xx = miSignalGenerator(sigType = "sine", vmax = vmax, dc = dc, ff = 200, nn = N, snr = 0)

XX1 = np.fft.fft(xx)
FF, XX2 = miDSFT(xx, fs = fs)
plt.figure(1, figsize = (12,4))
plt.clf()

plt.plot(FF, abs(XX2), '-', color = "magenta")
plt.plot(FF, abs(XX1), '--', color = "blue")

plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitud (V)')
plt.grid(linestyle='-', alpha=0.5)
plt.title(f'...')
plt.tight_layout()
plt.show()

#%% [markdown]
"""
Se puede corroborar que la diferencia entre ambas funciones es del orden de magnitud del error de punto flotante
"""

# %% 
XXERR = (abs(XX1)-abs(XX2))/abs(XX1)
plt.plot(FF, XXERR, '-', color = "red")

plt.xlabel('Freq (Hz)')
plt.ylabel('Error relativo')
plt.grid(linestyle='-', alpha=0.5)
plt.title(f'...')
plt.tight_layout()
plt.show()

# %%