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
fs = 2**20      #500Hz de BW
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
    
    if (snr != -1):
        Pna = pa / undB(snr)
        xx = xx + miNoise(Pot = Pna, nn = nn)

    return (tt, xx)

# %% [markdown]
"""

## Genero el bloque de cuantizacion

Cuando cuantizamos, dividimos el rango total en $2^B$ segmentos

Para que $-q/2 < e_q \leq q/2$, debemos redondear al centro de cada uno de esos escalones

```np.round``` redondea a enteros, por lo que ``` xx = np.round(xx/step) * step ``` siempre ubicara al 0 en el centro de los escalones

Para corregir esto, podemos restar un valor y volver a sumarlo luego de cuantizar. Es importante tener en cuenta que esto no afecta la señal en si, sino unicamente la ubicacion de los niveles de cuantizacion. Deseamos que al aplicar ```xx = np.round(xx/step) * step``` los nivel de cuantizacion se encuentren en $0\pm k \cdot q$

Para esto se resta vfs (ubicando a la señal entre 0 y vpp), luego restamos q/2.
"""

# %%

def miQuant(xx, B = B, vfsn = 0, vfsp = vfs):
    step = (vfsp - vfsn)/(2**B)
    xx = xx - vfsn - step/2 
    xx = np.clip(xx, a_min=0, a_max=vfsp-vfsn-step)
    xx = np.round(xx/step) * step
    xx = xx + vfsn + step/2

    return xx

#%% [markdown]
"""
## Test de la funcion

Para testear el bloque de cuantizacion generamos dos señales, una con ruido analogico y otra sin, cuantizamos ambas y comparamos los $e_q$
"""

# %% 

B=3
vfsn = -vfs
vfsp = vfs
step = (vfsp-vfsn)/(2**B)
tt, xx = miSignalGenerator(
    sigType = "sine",
    vmax = vmax,
    dc = 0.15,
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
    vmax = vmax,
    dc = 0.15,
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
pltSig.plot(tt, xxq, ':.', color = "magenta")
pltSig.hlines(y = vfsn, xmin = 0, xmax = 1, linestyle='--', color ="k")
pltSig.hlines(y = vfsp, xmin = 0, xmax = 1, linestyle='--', color ="k")
pltSig.set_title(f'Señal original (Azul) vs cuantizada con step de {step}V (Magenta); Sin ruido')

pltSigN.plot(tt, xxn, '-', color = "blue")
pltSigN.plot(tt, xxqn, ':.', color = "magenta")
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

##Analisis del ruido de cuantizacion

Segun la teoria, el ruido de cuantizacion deberia ser uniforme, tener media = 0, varianza = $\frac{q/2}{\sqrt(12)}$ y ser incorelacionado, cuando haya suficiente ruido analogico.

Para verificar esto tomamos ambas señales del bloque previo y graficamos los histogramas de sus ruidos. Para cada histograma se muestra la media y varianza ideal (negro) versus la real de la funcion (rojo)

Luego graficamos la varianza de ambas funciones


"""

# %% 

plt.clf()
plt.figure(figsize=(10, 8))

pltQNN = plt.subplot(2,2,1)
pltQN = plt.subplot(2,2,2, sharey=pltQNN)
pltAutocor = plt.subplot(2,1,2)

bins=40
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

pltQNN.set_title(f'Qn de una señal con\n ruido analogico')
pltQNN.set_xlabel('Eq [V]')
pltQNN.set_ylabel('Conteos')

pltQN.hist(qn, bins=bins)

pltQN.vlines(x = -step/2, ymin = 0, ymax = N/bins, linestyle='--', color ="k")
pltQN.hlines(y = N/bins, xmin = -step/2, xmax = step/2, linestyle='--', color ="k")
pltQN.vlines(x = step/2, ymin = 0, ymax = N/bins, linestyle='--', color ="k")

pltQN.axvline(x = 0, ymin = 0, ymax = 1, linestyle=':', color ="k")
pltQN.axvline(x = step/np.sqrt(12), ymin = 0, ymax = 1, linestyle='-.', color ="k")
pltQN.axvline(x = -step/np.sqrt(12), ymin = 0, ymax = 1, linestyle='-.', color ="k")

pltQN.axvline(x = np.average(qn), ymin = 0, ymax = 1, linestyle=':', color ="red")
pltQN.axvline(x = np.sqrt(np.var(qn)), ymin = 0, ymax = 1, linestyle='-.', color ="red")
pltQN.axvline(x = -np.sqrt(np.var(qn)), ymin = 0, ymax = 1, linestyle='-.', color ="red")

pltQN.set_title(f'Qn de una señal sin\n ruido analogico')
pltQN.set_xlabel('Eq [V]')
                  
rnn = sig.correlate(qnn, qnn)
rn = sig.correlate(qn, qn)

pltAutocor.plot(tt, rn[N-1:], '-', color="red")
pltAutocor.plot(tt, rnn[N-1:], '-', color="blue")

pltAutocor.set_title(f'Autocorrelacion de las señales con ruido analogico (Azul) y sin ruido Analogico (Rojo)', y=0)
pltAutocor.set_xlabel('Tiempo [S]')
pltAutocor.grid(linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
Podemos ver a simple vista que la señal con ruido analogico cumple con todas las condiciones, mientras que la señal sin ruido analogico no cumple con ninguna de estas (si cumpliria con la media de haber usado una señal original de dc nulo)
"""

#%% [markdown]
"""

## Un poco mas alla

Si no estamos conformes con un analisis "a simple vista" del ruido de cuantizacion, ¿que otras herramientas podes usar?

En esta seccion repetiremos los pasos previos para un snr variable, y para cada señal obtenida realizaremos un test de Kolmogorov–Smirnov para poder analizar que tan bien se adapta cada señal a una distribucion uniforme ideal

"""

#%%

snrs = np.arange(500)/10
sigmaarr = np.ones(len(snrs)) 
pvalarr = np.ones(len(snrs))
Darr = np.ones(len(snrs))
for i in range(len(snrs)):
  tt, xxarr = miSignalGenerator(
      sigType = "sine",
      vmax = vmax,
      dc = 0,
      ff = N/4,
      ph = 0,
      nn = N,
      fs = fs,
      snr = snrs[i])
  sigmaarr[i] = np.sqrt((vmax**2/2) / undB(snrs[i])) 
  xxarrq = miQuant(
    xxarr,
    B = B,
    vfsn = vfsn,
    vfsp = vfsp)
  qnarr = xxarrq - xxarr
  qnarr_scaled = (qnarr/step)+0.5
  Darr[i], pvalarr[i] = sta.kstest(qnarr_scaled, 'uniform')

# %% [markdown]

"""
nota: si bien se podria haber utilizado notacion matricial en vez de iterar un loop:

```py
  tt = np.arange(nn) * 1/fs
  xx= np.outer(np.ones(len(snrs)), (vmax * np.sin( 2 * np.pi * ff * tt)))
  na = np.random.normal(0, np.sqrt(Pot), (len(snrs), nn))
  na_amp = np.outer(((vmax**2/2) *10**(-snrs/10)), np.ones(nn))
  nac = na * na_amp
  xx = xx + nac/len(snrs)
```
en esta ocacion esto resulta computacionalmete mas costoso, ya que cada señal ocupa una cantidad no despreciable de memoria, y usar una matriz requiere reservar esa misma cantidad por cada fila, mientras que al iterar se reserva una unica vez y se sobreescribe la misma seccion de memoria. Para el tamaño de esta matriz, hubiese sido imposible reservar suficiente memoria (lo intente)

"""

# %% [markdown]

"""

### Ploteo de los resultados obtenidos

en los siguientes graficos podemos ver los resultados del K-S test (azul) y el pval de confiabilidad de esos resultados, tanto en funcion del snr como del sigma del ruido en si. Se marca el nivel de ruido para el cual la desviacion estandar del ruido es igual a medio escalon

"""

#%%

plt.clf()
plt.figure(figsize=(10, 8))

pval_snr = plt.subplot(2,1,1)
pval_ratio = plt.subplot(2,1,2)

D_snr = pval_snr.twinx()
D_ratio = pval_ratio.twinx()

k = 1
##snr para el cual sigma/(q/2) = k
pval_snr.axvline(x=20*np.log10(np.sqrt(2)*vmax/(step*k)),ymin=0,ymax=1, linestyle='--', color='purple')
pval_snr.axhline(y=0.05,xmin=0,xmax=1, linestyle='--', color='red')

pval_ratio.axvline(x=k,ymin=0,ymax=1, linestyle='--', color='purple')
pval_ratio.axhline(y=0.05,xmin=0,xmax=1, linestyle='--', color='red')

### SNR para el cual Pn = (step/2)**2
## es decir, sigma_n = (q/2)

pval_snr.plot(snrs, pvalarr, ':.', lw=1, alpha=0.8, color='red')
D_snr.plot(snrs, Darr, '-', lw=0.8, color='blue')

pval_ratio.plot(sigmaarr/(step/2), pvalarr, ':.', lw=1, alpha=0.8, color='red')
D_ratio.plot(sigmaarr/(step/2), Darr, '-', lw=0.8, color='blue')

pval_snr.set_title(f'D,pval vs SNR')
pval_snr.set_xlabel('SNR [dB]')
pval_snr.set_ylabel('pvalue')
D_snr.set_ylabel('D')

pval_snr.tick_params(axis='y', colors='red')
D_snr.tick_params(axis='y', colors='blue')
pval_snr.tick_params(axis='x', colors='purple')

pval_snr.grid(axis='both', linestyle='-', alpha=0.4, color='blue')


pval_ratio.set_title(f'D,pval vs desviacion estandard del ruido analogico expresado como fraccion del escalon de cuantizacion')
pval_ratio.set_xlabel('sigma/(q/2)')
pval_ratio.set_ylabel('pvalue')
D_ratio.set_ylabel('D')

pval_ratio.tick_params(axis='y', colors='red')
D_ratio.tick_params(axis='y', colors='blue')
pval_ratio.tick_params(axis='x', colors='purple')

pval_ratio.grid(axis='both', linestyle='-', alpha=0.4, color='blue')




plt.tight_layout()
plt.show()

# %% [markdown]
"""
En el grafico anterior se puede ver como aumenta la desviacion de la uniformidad al aumentar el snr de la funcion. puede notarse tambien un cumulo de valores para los cuales la desviacion es tan pequeña que deja de ser estadisticamente detectable
"""
# %%
