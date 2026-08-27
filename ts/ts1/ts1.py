# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.14.6
# ---

# %% [markdown]
"""
# Tarea Semanal 1
    la amplitud máxima de la senoidal (volts)
    su valor medio (volts)
    la frecuencia (Hz)
    la fase (radianes)
    la cantidad de muestras digitalizada por el ADC (# muestras)
    la frecuencia de muestreo del ADC.

es decir que la función que uds armen debería admitir se llamada de la siguiente manera

tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)
"""

# %% Import execution={"iopub.execute_input": "2026-08-27T19:45:35.845888Z", "iopub.status.busy": "2026-08-27T19:45:35.845652Z", "iopub.status.idle": "2026-08-27T19:45:36.906418Z", "shell.execute_reply": "2026-08-27T19:45:36.905860Z"}

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# %% General Defs execution={"iopub.execute_input": "2026-08-27T19:45:36.908499Z", "iopub.status.busy": "2026-08-27T19:45:36.908286Z", "iopub.status.idle": "2026-08-27T19:45:36.910750Z", "shell.execute_reply": "2026-08-27T19:45:36.910348Z"}

def undB (dB):
    return 10**(dB/10)

def dB (x):
    return 10*np.log10(x)

# %% Params Generales execution={"iopub.execute_input": "2026-08-27T19:45:36.912537Z", "iopub.status.busy": "2026-08-27T19:45:36.912401Z", "iopub.status.idle": "2026-08-27T19:45:36.915079Z", "shell.execute_reply": "2026-08-27T19:45:36.914423Z"}
fs = 1000       #500Hz de BW
ts = 1/fs
N = fs          #DeltaF = 1Hz; norm
df = fs/N
dt = 1/df

# %% Params func execution={"iopub.execute_input": "2026-08-27T19:45:36.916960Z", "iopub.status.busy": "2026-08-27T19:45:36.916830Z", "iopub.status.idle": "2026-08-27T19:45:36.919464Z", "shell.execute_reply": "2026-08-27T19:45:36.918605Z"}
vmax = 1.        
dc = 0.          
ff = 5
ph = 0

snr_xr = 30

# %% Func Defs execution={"iopub.execute_input": "2026-08-27T19:45:36.920814Z", "iopub.status.busy": "2026-08-27T19:45:36.920674Z", "iopub.status.idle": "2026-08-27T19:45:36.930567Z", "shell.execute_reply": "2026-08-27T19:45:36.929681Z"}
def miNoise (Pot, nn):
    xna = np.random.normal(loc = 0, scale = np.sqrt(Pot), size = N) 
    return xna

def miSin (vmax, dc, ff, ph, nn, fs):
    tt = np.arange(N) * 1/fs
    xx = dc + vmax * np.sin( 2 * np.pi * ff * tt + ph)
    return (tt, xx)

def miSaw (vmax, dc, ff, ph, nn, fs):
    tt = np.arange(N) * 1/fs
    To = 1/ff
    delay = To * ph/(2*np.pi)
    ####
    xx =  dc - (vmax / 2) + vmax * ((tt + delay)% To) * ff
    return (tt, xx)


def miTri (vmax, dc, ff, ph, nn, fs):
    tt = np.arange(N) * 1/fs
    To = 1/ff
    delay = To * ph/(2*np.pi)
    ####    
    xx =  2 * vmax * ((tt + delay)% To) * ff
    for i in range(len(tt)):    
        if (0.5 <= ff*((tt[i] + To * ph/(2*np.pi)) % To)):
            xx[i] =  2 * vmax - xx[i]
    xx = xx + dc - (vmax/4)
    return (tt, xx)
    
def miPWM (vmax, dc, ff, ph, nn, fs, duty):
    tt = np.arange(N) * 1/fs
    ####    
    xx = np.arange(N)
    for i in range(len(tt)):
        if (duty < ((tt[i]*ff + ph/(2*np.pi)) % 1)):
            xx[i] = vmax
        else:
            xx[i] = 0
    xx = xx + dc - (vmax * duty)

def miSignalGenerator (sigType = "sine", vmax = vmax, dc = dc, ff = ff, ph = ph, nn = N, fs = fs, snr = -1, duty = 0.5):
    match sigType:
        case "sine":
            tt, xx = miSin(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs)
        case "saw":
            tt, xx = miSaw(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs)
        case "revSaw":
            tt, xx = miSaw(vmax = -vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs)
        case "tri":
            tt, xx = miTri(vmax = -vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs)
        case "PWM":
            tt, xx = miPWM(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs, duty = duty)
        case "square":
            tt, xx = miPWM(vmax = vmax, dc = dc, ff = ff, ph = ph, nn = nn, fs = fs, duty = 0.5)
        case _:
            print("sigType invlaido")
            tt = np.arange(N)
            xx = np.arange(N)
    
    if (snr != -1):
        Px = vmax**2/2
        Pna = Px / undB(snr)
        xx = xx + miNoise(Pot = Pna, nn = nn)
    return (tt, xx)


# %% execution={"iopub.execute_input": "2026-08-27T19:45:36.932535Z", "iopub.status.busy": "2026-08-27T19:45:36.932352Z", "iopub.status.idle": "2026-08-27T19:45:37.093834Z", "shell.execute_reply": "2026-08-27T19:45:37.092939Z"}
tt, xx = miSignalGenerator(dc = 0.5, snr = 20)

plt.figure(1, figsize = (12,4))
plt.clf()
           
plt.plot(tt,xx, linewidth=1.5)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.grid(linestyle='-', alpha=0.5)
plt.title(f'Sinusoidal de {ff} Hz')
plt.tight_layout()
plt.show()

# %% execution={"iopub.execute_input": "2026-08-27T19:45:37.095482Z", "iopub.status.busy": "2026-08-27T19:45:37.095321Z", "iopub.status.idle": "2026-08-27T19:45:37.201693Z", "shell.execute_reply": "2026-08-27T19:45:37.200834Z"}
tt2, xx2 = miSignalGenerator("tri", dc = 0.5, snr = 20)

plt.figure(2, figsize = (12,4))
plt.clf()
           
plt.plot(tt,xx, linewidth=1.5)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.grid(linestyle='-', alpha=0.5)
plt.title(f'Sinusoidal de {ff} Hz')
plt.tight_layout()
plt.show()
