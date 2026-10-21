# %% [markdown]
"""
"""

# %% [markdown]
"""
"""

# %% Import and directives

import numpy as np
import cmath as cm

import matplotlib.pyplot as plt
import matplotlib.patches as ptch

import scipy.signal as sig
import scipy.stats as sta
import scipy.io as sio

from collections import Counter
#!%matplotlib qt

#%% [markdown]
"""
"""

# %% General Defs 

def undB (dB):
    return 10**(dB/10)

def dB (x):
    return 10*np.log10(x)

#%% [markdown]
"""
"""

# %% 

j = complex(0, 1)

#%% [markdown]
"""
"""


#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo io.whosmat('ECG_TP4.mat')

mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

#ecg_one_lead = np.load('ecg_sin_ruido.npy')

#plt.figure()
#plt.plot(ecg_one_lead)



#%% [markdown]
"""
IIR
"""

wp=35
ws=36
gpass=-3
gstop=-100
worN = 2**16
filters=['butter', 'cheby1', 'ellip', 'cheby2']
cols=['blue', 'green', 'orange', 'purple']

sos = {}
FF = {}
XX = {}
delay = {}
z = {}
p = {}
k = {}

for i, filt in enumerate(filters):
  sos[filt] = sig.iirdesign(wp=wp, ws=ws, gpass=-gpass, gstop=-gstop, analog=False, ftype=filt, output='sos', fs=fs_ecg)
  FF[i,:], XX[i,:] = sig.freqz_sos(sos[filt], worN=worN, fs=fs_ecg)
  delay[i,:]=np.unwrap(np.angle(XX[i,:]))[1:len(XX[i,:])] - np.unwrap(np.angle(XX[i,:]))[0:len(XX[i,:])-1]
  z[i,:], p[i,:], k[i,:] = sig.sos2zpk(sos[filt])



# %%
fig, (axAmp, axDel) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axPha = axAmp.twinx()

minf=0
maxf=80
ming=-150
maxg=10
mind=-500
maxd=100

for i, filt in enumerate(filters):
  axAmp.plot(FF[i,:], 20 * np.log10(np.abs(XX[i,:])), color=cols[i])
  axPha.plot(FF[i,:], np.unwrap(np.angle(XX[i,:])), ls='--', color=cols[i])
  axDel.plot(FF[i,:][:len(XX[1,:])-1], 1000*delay[i,:], color=cols[i], label=filt)
  
axAmp.set_ylabel('Magnitud [dB]')
axPha.set_ylabel('Phase [rad]')
axDel.set_xlabel('Freq [Hz]')
axDel.set_ylabel('Delay [ms]')

axDel.grid(True, which='both', alpha=0.4)
axAmp.grid(True, which='both', alpha=0.4)

axAmp.set_xlim(minf, maxf)
axAmp.set_ylim(ming, maxg)
axDel.set_ylim(mind, maxd)
axAmp.set_title('IIR Pasabajos Multiples Topologias')
axDel.legend(loc='upper left')

if(wp < ws):
  axAmp.add_patch(ptch.Rectangle((wp, gpass), minf-wp, ming-gpass,
                       facecolor='red', alpha=0.15, hatch='x',
                       edgecolor='red', lw=1.5))

  axAmp.add_patch(ptch.Rectangle((ws, gstop), maxf-ws, maxg-gstop,
                       facecolor='red', alpha=0.15, hatch='x',
                       edgecolor='red', lw=1.5))

if(ws < wp):
  axAmp.add_patch(ptch.Rectangle((wp, gpass), maxf-wp, maxg-gpass,
                       facecolor='red', alpha=0.15, hatch='x',
                       edgecolor='red', lw=1.5))

  axAmp.add_patch(ptch.Rectangle((ws, gstop), minf-ws, ming-gstop,
                       facecolor='red', alpha=0.15, hatch='x',
                       edgecolor='red', lw=1.5))

plt.tight_layout()
plt.show()

# %%

fig, ax = plt.subplots(figsize=(6, 6))

# Unit circle
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)

# Zeros (circles) and poles (crosses)
for i, filt in enumerate(filters):
  ax.plot(z[i,:].real, z[i,:].imag, 'o', mfc='none', ms=9, label=f'{filt} ({len(p[i,:])},{len(z[i,:])})', color=cols[i])
  ax.plot(p[i,:].real, p[i,:].imag, 'x', ms=9, mew=2, color=cols[i])
  for val, n in Counter(np.round(z[i,:], 6)).items():
    if n > 1:
      ax.annotate(f'{n}z', (val.real, val.imag), textcoords='offset points', xytext=(10, 10*i), color=cols[i])
  for val, n in Counter(np.round(p[i,:], 6)).items():
    if n > 1:
      ax.annotate(f'{n}p', (val.real, val.imag), textcoords='offset points', xytext=(10, -10*i), color=cols[i])

ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_ylabel('Imaginary')
ax.set_title('Diagramas de Polos y ceros')
ax.grid(True, alpha=0.4)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()


#%% [markdown]
"""

"""
