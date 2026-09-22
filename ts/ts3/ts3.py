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
fs = 2**20      #500Hz de BW
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
    vmax = 0.85,
    dc = 0,
    ff = 4,
    nn = N,
    snr = -1)
xxq = miQuant(
    xx,
    vfsp = vfsp,
    vfsn = vfsn,
    B = B)
tt, xxn = miSignalGenerator(
    sigType = "sine",
    vmax = 0.85,
    dc = 0,
    ff = 4,
    nn = N,
    snr = 10)
xxqn = miQuant(
    xxn,
    vfsp = vfsp,
    vfsn = vfsn,
    B = B)
qn = xxq - np.clip(xx, vfsn, vfsp)
qnn = xxqn - np.clip(xxn, vfsn, vfsp)
      #si xx se pasa de los limites de vfs ese error deberia ignorarse porque no es ruido de cuantizacion

plt.clf()
fig, (pltSig, pltSigN, pltErr) = plt.subplots(
  3, 1,
  figsize=(12, 6),
  sharex=True,
  gridspec_kw={"height_ratios": [3, 3, 2]}
  )

pltSig.plot(tt, xx, '-', color = "blue")
pltSig.plot(tt, xxq, 'x--', color = "magenta")
pltSig.hlines(y = vfsn, xmin = 0, xmax = 1, linestyle='--', color ="k")
pltSig.hlines(y = vfsp, xmin = 0, xmax = 1, linestyle='--', color ="k")
pltSig.set_title(f'Señal original (Azul) vs cuantizada con step de {step}V (Magenta); Sin ruido')

pltSigN.plot(tt, xxn, '-', color = "blue")
pltSigN.plot(tt, xxqn, 'x--', color = "magenta")
pltSigN.hlines(y = vfsn, xmin = 0, xmax = 1, linestyle='--', color ="k")
pltSigN.hlines(y = vfsp, xmin = 0, xmax = 1, linestyle='--', color ="k")
pltSigN.set_title(f'Señal original (Azul) vs cuantizada con step de {step}V (Magenta); Con ruido')

pltErr.plot(tt, qnn, '-', color = "blue")
pltErr.plot(tt, qn, '-', color = "red")
pltErr.hlines(y = step/2, xmin = 0, xmax = N/fs, linestyle='--', color ="k")
pltErr.hlines(y = -step/2, xmin = 0, xmax = N/fs, linestyle='--', color ="k")
pltErr.set_title(f'Ruido de cuantizacion; Señal analogica con ruido (Azul), vs sin ruido (Rojo)')

plt.xlabel('Tiempo (s)')
pltErr.set_ylabel('Amplitud (V)')
pltSig.set_ylabel('Amplitud (V)')
pltSigN.set_ylabel('Amplitud (V)')
plt.grid(linestyle='-', alpha=0.5)
plt.tight_layout()

plt.show()

#%% [markdown]
"""
"""

# %% 

plt.clf()
plt.figure(figsize=(10, 8))

pltQNN = plt.subplot(2,2,1)
pltQN = plt.subplot(2,2,2, sharey=pltQNN)
pltAutocor = plt.subplot(2,1,2)

bins=20
pltQNN.hist(qnn, bins=bins)

pltQNN.vlines(x = -step/2, ymin = 0, ymax = N/bins, linestyle='--', color ="k")
pltQNN.hlines(y = N/bins, xmin = -step/2, xmax = step/2, linestyle='--', color ="k")
pltQNN.vlines(x = step/2, ymin = 0, ymax = N/bins, linestyle='--', color ="k")

pltQNN.axvline(x = 0, ymin = 0, ymax = 1, linestyle=':', color ="k")
pltQNN.axvline(x = step/np.sqrt(12), ymin = 0, ymax = 1, linestyle='-.', color ="k")
pltQNN.axvline(x = -step/np.sqrt(12), ymin = 0, ymax = 1, linestyle='-.', color ="k")

pltQNN.axvline(x = np.average(qnn), ymin = 0, ymax = 1, linestyle=':', color ="red")
pltQNN.axvline(x = np.sqrt(np.var(qnn)), ymin = 0, ymax = 1, linestyle='-.', color ="red")
pltQNN.axvline(x = -np.sqrt(np.var(qnn)), ymin = 0, ymax = 1, linestyle='-.', color ="red")

D, pvaln = sta.kstest((qnn/step)+0.5, 'uniform')
pltQNN.set_title(f'Qn de una señal con\n ruido analogico (K-S con p = {pvaln:.4e})')
pltQNN.set_xlabel('Eq [V]')
pltQNN.set_ylabel('Conteos')

pltQN.hist(qn, bins=20)

pltQN.vlines(x = -step/2, ymin = 0, ymax = N/bins, linestyle='--', color ="k")
pltQN.hlines(y = N/bins, xmin = -step/2, xmax = step/2, linestyle='--', color ="k")
pltQN.vlines(x = step/2, ymin = 0, ymax = N/bins, linestyle='--', color ="k")

pltQN.axvline(x = 0, ymin = 0, ymax = 1, linestyle=':', color ="k")
pltQN.axvline(x = step/np.sqrt(12), ymin = 0, ymax = 1, linestyle='-.', color ="k")
pltQN.axvline(x = -step/np.sqrt(12), ymin = 0, ymax = 1, linestyle='-.', color ="k")

pltQN.axvline(x = np.average(qn), ymin = 0, ymax = 1, linestyle=':', color ="red")
pltQN.axvline(x = np.sqrt(np.var(qn)), ymin = 0, ymax = 1, linestyle='-.', color ="red")
pltQN.axvline(x = -np.sqrt(np.var(qn)), ymin = 0, ymax = 1, linestyle='-.', color ="red")

D, pval = sta.kstest((qn/step)+0.5, 'uniform')
pltQN.set_title(f'Qn de una señal sin\n ruido analogico (K-S con p = {pval:.4e})')
pltQN.set_xlabel('Eq [V]')
                  
rnn = sig.correlate(qnn, qnn)
rn = sig.correlate(qn, qn)

pltAutocor.plot(tt, rn[N-1:], '-', color="red")
pltAutocor.plot(tt, rnn[N-1:], '-', color="blue")

pltAutocor.set_title(f'Autocorrelacion de las señales con ruido analogico (Azul) y sin ruido Analogico (Rojo)', y=0)
pltAutocor.set_xlabel('Tiempo [S]')
pltAutocor.grid(linestyle='-', alpha=0.5)

plt.tight_layout()
plt.s
#%% [markdown]
"""

"""
