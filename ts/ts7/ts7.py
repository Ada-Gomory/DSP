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
import matplotlib.gridspec as grd

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

def plotSOS(sos, labels, fs, limits, plant, colors=[]):
  #################################
  ## fft, dely, pzk
  #################################
  wp = plant[0]
  ws = plant[1]
  gpass = plant[2]
  gstop = plant[3]

  FF = {}
  XX = {}
  delay = {}
  z = {}
  p = {}
  k = {}

  for i, filt in enumerate(labels):
    FF[i,:], XX[i,:] = sig.freqz_sos(sos[filt], worN=2**18, fs=fs)
    delay[i,:]=np.unwrap(np.angle(XX[i,:]))[0:len(XX[i,:])-1]-np.unwrap(np.angle(XX[i,:]))[1:len(XX[i,:])]
    if not np.isfinite(sos[filt]).all():
      z[i,:], p[i,:], k[i,:] = sig.iirdesign(wp=wp, ws=ws, gpass=-gpass, gstop=-gstop, analog=False, ftype=filt, output='zpk', fs=fs_ecg)
    else:
      z[i,:], p[i,:], k[i,:] = sig.sos2zpk(sos[filt]) 
    
  #################################
  ## get color list if not given
  #################################
  if (len(colors) == 0):
    prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [prop_colors[i % len(prop_colors)] for i in range(len(labels))]

  #################################
  ## sublots and gridspec
  #################################
  fig = plt.figure(figsize=(15, 10))  

  gs = grd.GridSpec(4, 2, figure=fig, width_ratios=[2, 1])  

  axAmp = fig.add_subplot(gs[0:2, 0])
  axDel = fig.add_subplot(gs[2:4, 0], sharex=axAmp)
  axPzs = fig.add_subplot(gs[1:3, 1])
  axLeg = fig.add_subplot(gs[0, 1])
  axLeg.axis('off') 

  axPha = axAmp.twinx() 

  #################################
  ## plot
  #################################
  for i, filt in enumerate(labels):
    axAmp.plot(FF[i,:], 20 * np.log10(np.abs(XX[i,:])), color=colors[i])
    axPha.plot(FF[i,:], np.unwrap(np.angle(XX[i,:])), ls='--', color=colors[i])
    axDel.plot(FF[i,:][:len(XX[1,:])-1], 1000*delay[i,:], color=colors[i], label=filt)  

  theta = np.linspace(0, 2 * np.pi, 400)
  axPzs.plot(np.cos(theta), np.sin(theta), 'k--', lw=1) 

  for i, filt in enumerate(labels):
    axPzs.plot(z[i,:].real, z[i,:].imag, 'o', mfc='none', ms=9, label=f'{filt} ({len(p[i,:])},{len(z[i,:])})', color=colors[i])
    axPzs.plot(p[i,:].real, p[i,:].imag, 'x', ms=9, mew=2, color=colors[i])
    for val, n in Counter(np.round(z[i,:], 6)).items():
      if n > 1:
        axPzs.annotate(f'{n}z', (val.real, val.imag), textcoords='offset points', xytext=(10, 10*i), color=colors[i])
    for val, n in Counter(np.round(p[i,:], 6)).items():
      if n > 1:
        axPzs.annotate(f'{n}p', (val.real, val.imag), textcoords='offset points', xytext=(10, -10*i), color=colors[i])  

  #################################
  ## plot limits
  #################################
  minf=limits[0]
  maxf=limits[1]
  ming=limits[2]
  maxg=limits[3]
  maxd=0

  for i, filt in enumerate(labels):
    if(np.max(1000*delay[i,:]) > maxd):
      maxd = np.max(1000*delay[i,:])
  maxd = maxd * 1.1
  mind = -0.4 * maxd  

  axAmp.set_xlim(minf, maxf)
  axAmp.set_ylim(ming, maxg)
  axDel.set_ylim(mind, maxd)  

  #################################
  ## axlines and patch hatch
  #################################
  if(len(ws) == 1):
    if(wp < ws): #lowpass
      axAmp.add_patch(ptch.Rectangle((wp, gpass), minf-wp, ming-gpass,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  

      axAmp.add_patch(ptch.Rectangle((ws, gstop), maxf-ws, maxg-gstop,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  

    if(ws < wp): #highpass
      axAmp.add_patch(ptch.Rectangle((wp, gpass), maxf-wp, maxg-gpass,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  
  
      axAmp.add_patch(ptch.Rectangle((ws, gstop), minf-ws, ming-gstop,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))

  if(len(ws) == 2):
    if(wp[0] < ws[0]): #bandpass
      axAmp.add_patch(ptch.Rectangle((wp[0], gpass), minf-wp[0], ming-gpass,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  
      
      axAmp.add_patch(ptch.Rectangle((ws[0], gstop), ws[1]-ws[0], maxg-gstop,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  

      axAmp.add_patch(ptch.Rectangle((wp[1], gpass), maxf-wp[1], ming-gpass,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  

    if(ws[0] < wp[0]): #bandpass
      axAmp.add_patch(ptch.Rectangle((ws[0], gstop), minf-ws[0], maxg-gstop,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  
      
      axAmp.add_patch(ptch.Rectangle((wp[0], gpass), wp[1]-wp[0], ming-gpass,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  

      axAmp.add_patch(ptch.Rectangle((ws[1], gstop), maxf-ws[1], maxg-gstop,
                           facecolor='red', alpha=0.15, hatch='x',
                           edgecolor='red', lw=1.5))  
                           

  axPzs.axhline(0, color='gray', lw=0.5)
  axPzs.axvline(0, color='gray', lw=0.5)  

  #################################
  ## labels, grid, title, legends
  #################################
  axAmp.set_ylabel('Magnitud [dB]')
  axPha.set_ylabel('Phase [rad]')
  axDel.set_xlabel('Freq [Hz]')
  axDel.set_ylabel('Delay [ms]')  

  axAmp.set_xscale('log')

  axDel.grid(True, which='both', alpha=0.4)
  axAmp.grid(True, which='both', alpha=0.4)
  axPzs.grid(True, alpha=0.4) 

  axAmp.set_title('IIR Pasabajos Multiples Topologias')
  axPzs.set_title('Diagramas de Polos y ceros') 

  handles, labels = axPzs.get_legend_handles_labels()
  axLeg.legend(handles, labels, loc='center', title='Filter type')  

  #################################
  ## show
  #################################
  plt.tight_layout()
  plt.show()


#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = (mat_struct['ecg_lead']*1.0).flatten()
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

#%% [markdown]
"""
IIR
"""

sos = {}

filters=['butter', 'cheby1', 'cheby2', 'ellip']
colors=['red', 'blue', 'green', 'purple', 'orange']

wp=[1, 35]
ws=[0.4, 50]
gpass=-3
gstop=-400

for i, filt in enumerate(filters):
  sos[filt] = sig.iirdesign(wp=wp, ws=ws, gpass=-gpass, gstop=-gstop, analog=False, ftype=filt, output='sos', fs=fs_ecg)

plotSOS(sos=sos, labels=filters, fs=fs_ecg,
        limits=[0.05, 80, -500, 10],
        plant=[wp, ws, gpass, gstop],
        colors=colors[1:])

#%% [markdown]
"""
Filtering
"""

ecg_filtered = {}
for i, filt in enumerate(filters):
  ecg_filtered[i] = sig.sosfiltfilt(sos[filt], ecg_one_lead)

tt = np.arange(len(ecg_one_lead)) * 1/fs_ecg

#################################
## sublots and gridspec
#################################
fig = plt.figure(figsize=(15, 10))  
gs = grd.GridSpec(1, 1, figure=fig) #, width_ratios=[2, 1])  
axEcg = fig.add_subplot(gs[0, 0])

#################################
## plot
#################################
axEcg.plot(tt, ecg_one_lead, color='red', label='none')
for i, filt in enumerate(filters):
  axEcg.plot(tt, ecg_filtered[i], color=colors[i+1], label=filters[i])

#################################
## labels, grid, title, legends
#################################
axEcg.set_ylabel('A')
axEcg.set_xlabel('Tiempo [s]')

axEcg.grid(True, which='both', alpha=0.4)
axEcg.grid(True, which='both', alpha=0.4)
axEcg.grid(True, alpha=0.4) 

axEcg.set_title('Ecg con y sin filtrar')
axEcg.legend(loc='lower right') 

#################################
## show
#################################
plt.tight_layout()
plt.show()


#%% [markdown]
"""

"""

# %%
