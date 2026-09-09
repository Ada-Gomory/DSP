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
fs = 1024       #500Hz de BW
ts = 1/fs
N = fs        #DeltaF = 1Hz; norm
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

def miSignalGenerator (sigType = "sqr", vmax = vmax, dc = dc, ff = ff, ph = ph, nn = N, fs = fs, snr = -1, duty = 0.5, B = 0, vfsp = vfs, vfsn = 0):
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

def miQuant(xx, B = B, vfsn = 0, vfsp = vfs):
    if (B > 0):
        step = (vfsp - vfsn)/(2**B)
        xx = xx - step/2 
        xx = np.clip(xx, a_min=vfsn, a_max=vfsp-step)
        xx = np.round(xx/step) * step
        xx = xx + step/2

    return xx

#%% [markdown]
"""
## Test de la funcion

Se utiliza la funcion de fft provista por numpy en azul punteado para corroborar que el resultado es identico al esperado
"""

# %% 

B=3
vfsn = -vfs
vfsp = vfs
step = (vfsp-vfsn)/(2**B)
tt, xx = miSignalGenerator(
    sigType = "sine",
    vmax = 1.25,
    dc = 0,
    ff = 5,
    nn = N,
    snr = 20)
xxq = miQuant(
    xx,
    vfsp = vfsp,
    vfsn = vfsn,
    B = B)
qn = xx - xxq

plt.clf()
fig, (pltSig, pltErr) = plt.subplots(
  2, 1,
  figsize=(12, 6),
  sharex=True,
  gridspec_kw={"height_ratios": [3, 2]}
  )

pltSig.plot(tt, xx, '-', color = "blue")
pltSig.plot(tt, xxq, 'x--', color = "magenta")
pltSig.axhline(y = vfsn, xmin = 0, xmax = N/fs, linestyle='--', color ="k")
pltSig.axhline(y = vfsp, xmin = 0, xmax = N/fs, linestyle='--', color ="k")

pltErr.plot(tt, qn, '-', color = "red")
pltErr.axhline(y = step/2, xmin = 0, xmax = N/fs, linestyle='--', color ="k")
pltErr.axhline(y = -step/2, xmin = 0, xmax = N/fs, linestyle='--', color ="k")

plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.grid(linestyle='-', alpha=0.5)
plt.title(f'Señal original (Azul) vs cuantizada con step de {step}V (Magenta)')
plt.tight_layout()

plt.show()


#%% [markdown]
"""
"""

# %% 

XX = np.fft.fft(xx)
XXQ = np.fft.fft(xxq)
FF = np.arange(len(XX)) *  fs/len(XX) 

plt.figure(1, figsize = (12,4))
plt.clf()

plt.plot(FF, abs(XX), '-', color = "blue")
plt.plot(FF, abs(XXQ), 'x--', color = "magenta")

plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitud (V)')
plt.grid(linestyle='-', alpha=0.5)
plt.title(f'...')
plt.tight_layout()
plt.show()

#%% [markdown]
"""
"""
