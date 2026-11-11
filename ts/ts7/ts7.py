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

def toMatrix(rows, dtype=complex):
  #Stack 1D arrays of different lengths into a matrix, padding with NaN.
  m = np.full((len(rows), max(len(r) for r in rows)), np.nan, dtype=dtype)
  for i, r in enumerate(rows):
    m[i, :len(r)] = r
  return m

def plotFilt(FF, XX, delay, z, p, labels, limits, plant, colors=[]):
  #################################
  ## fft, dely, pzk
  #################################
  wp = plant[0]
  ws = plant[1]
  gpass = plant[2]
  gstop = plant[3]

    
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
  axLeg.axis('off') #Used to put legends outside of graphs

  axPha = axAmp.twinx() 

  #################################
  ## Plot Bode
  #################################
  for i, filt in enumerate(labels):
    axAmp.plot(FF[i,:], 20 * np.log10(np.abs(XX[i,:])), color=colors[i])
    axPha.plot(FF[i,:], np.unwrap(np.angle(XX[i,:])), ls='--', color=colors[i])
    axDel.plot(FF[i,:][:len(XX[i,:])-1], 1000*delay[i,:], color=colors[i], label=filt)  

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
  boxes = []   # (xmin, xmax, ymin, ymax) of every forbidden region

  def addBox(x, y, w, h):
    axAmp.add_patch(ptch.Rectangle((x, y), w, h,
                         facecolor='k', alpha=0.2, hatch='x',
                         edgecolor='k', lw=1.5))
    # widths/heights can be negative, so normalise the bounds
    boxes.append((min(x, x + w), max(x, x + w),
                  min(y, y + h), max(y, y + h)))

  if(len(ws) == 1):
    if(wp < ws): #lowpass
      addBox(wp, -gpass, minf-wp, ming+gpass)
      addBox(ws, -gstop, maxf-ws, maxg+gstop)

    if(ws < wp): #highpass
      addBox(wp, -gpass, maxf-wp, maxg+gpass)
      addBox(ws, -gstop, minf-ws, ming+gstop)

  if(len(ws) == 2):
    if(wp[0] < ws[0]): #bandstop
      addBox(wp[0], -gpass, minf-wp[0], ming+gpass)
      addBox(ws[0], -gstop, ws[1]-ws[0], maxg+gstop)
      addBox(wp[1], -gpass, maxf-wp[1], ming+gpass)

    if(ws[0] < wp[0]): #bandpass
      addBox(ws[0], -gstop, minf-ws[0], maxg+gstop)
      addBox(wp[0], -gpass, wp[1]-wp[0], ming+gpass)
      addBox(ws[1], -gstop, maxf-ws[1], maxg+gstop)

  axPzs.axhline(0, color='gray', lw=0.5)
  axPzs.axvline(0, color='gray', lw=0.5)  

  #################################
  ## red x where a curve touches a forbidden box
  #################################  
  errorCounts = np.zeros(len(labels), dtype=int)

  for i, filt in enumerate(labels):
    f = FF[i,:]
    with np.errstate(divide='ignore'):
      mag = 20 * np.log10(np.abs(XX[i,:]))

    inside = np.zeros(len(f), dtype=bool)
    for (x0, x1, y0, y1) in boxes:
      inside |= (f >= x0) & (f <= x1) & (mag >= y0) & (mag <= y1)

    # first and last sample of each contiguous run inside a box
    edges = np.diff(np.concatenate(([0], inside.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1) - 1
    idx = np.unique(np.concatenate((starts, ends)))

    errorCounts[i] = len(idx)
    axAmp.plot(f[idx], mag[idx], 'x', color=colors[i], ms=8, mew=2, zorder=5)

  #################################
  ## Plot pzk
  #################################
  theta = np.linspace(0, 2 * np.pi, 400)
  axPzs.plot(np.cos(theta), np.sin(theta), 'k--', lw=1) 

  for i, filt in enumerate(labels):
    zi = z[i][~np.isnan(z[i])]     # drop padding
    pk = p[i][~np.isnan(p[i])]

    axPzs.plot(zi.real, zi.imag, 'o', mfc='none', ms=9, label=f'{filt} (Ord={len(pk)}; {errorCounts[i]} Errors)', color=colors[i])
    axPzs.plot(pk.real, pk.imag, 'x', ms=9, mew=2, color=colors[i])
    for val, n in Counter(np.round(zi, 6)).items():
      if n > 1:
        axPzs.annotate(f'{n}z', (val.real, val.imag), textcoords='offset points', xytext=(10, 10*i), color=colors[i])
    for val, n in Counter(np.round(pk, 6)).items():
      if n > 1:
        axPzs.annotate(f'{n}p', (val.real, val.imag), textcoords='offset points', xytext=(10, -10*i), color=colors[i])  

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

  axAmp.set_title('Bodes')
  axPzs.set_title('Diagramas de Polos y ceros') 

  handles, labels = axPzs.get_legend_handles_labels()
  axLeg.legend(handles, labels, loc='center', title='Filter type')  

  #################################
  ## show
  #################################
  plt.tight_layout()
  plt.show()

def plotSOS(sos, labels, fs, plant, worN):
  wp = plant[0]
  ws = plant[1]
  gpass = plant[2]
  gstop = plant[3]

  FF, XX, delay, z, p = [], [], [], [], []

  for filt in labels:
    ffi, xxi = sig.freqz_sos(sos[filt], worN=worN, fs=fs)
    phi = np.unwrap(np.angle(xxi))
    di = phi[:-1] - phi[1:]
    if not np.isfinite(sos[filt]).all():
      zi, pi, ki = sig.iirdesign(wp=wp, ws=ws, gpass=gpass, gstop=gstop, analog=False, ftype=filt, output='zpk', fs=fs)
    else:
      zi, pi, ki = sig.sos2zpk(sos[filt])

    FF.append(ffi)
    XX.append(xxi)
    delay.append(di)
    z.append(zi)
    p.append(pi)

  return FF, XX, delay, z, p

def plotFIR(firs, labels, fs, worN):

  FF, XX, delay, z, p, k = [], [], [], [], [], []

  for win in labels:
    ffi, xxi = sig.freqz(firs[win], a=1, worN=worN, fs=fs)
    phi = np.unwrap(np.angle(xxi))
    di = phi[:-1] - phi[1:]

    FF.append(ffi)
    XX.append(xxi)
    delay.append(di)
    z.append(np.roots(firs[win]))              # zeros: roots of the tap polynomial
    p.append(np.zeros(len(firs[win]) - 1))     # poles: all at z = 0

  return FF, XX, delay, z, p

  
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

colors=['red', 'blue', 'green', 'purple', 'orange', 'hotpink', 'teal']

"""
IIR
"""

sos = {}
filters=['butter', 'cheby1', 'cheby2', 'ellip']

wp=[1, 35]
ws=[0.4, 50]
gpass=1
gstop=40

for i, filt in enumerate(filters):
  sos[filt] = sig.iirdesign(wp=wp, ws=ws, gpass=gpass, gstop=gstop, analog=False, ftype=filt, output='sos', fs=fs_ecg)


"""
FIR
"""

firs = {}
windows=['boxcar', 'hamming']

taps = 1500

delta_w = 0.5
freq_win = [0., ws[0]+delta_w, wp[0], wp[1], wp[1]+1, fs_ecg/2]
gain_win = [ 0,             0,     1,     1,       0,        0]

for i, win in enumerate(windows):
  firs[win] = sig.firwin2(taps, freq_win, gain_win, window=win, fs=fs_ecg)
  
fff, xxf, df, zf, pf = plotFIR(firs=firs, labels=windows,
                              fs=fs_ecg, worN=2**18)

ffs, xxs, ds, zs, ps = plotSOS(sos=sos, labels=filters,
                              fs=fs_ecg, worN = 2**18,
                              plant=[wp, ws, gpass, gstop])

#plotFir and plotSos return lists of arrays, plot Filt expexts matrices
plotFilt(FF=np.array(ffs+fff), XX=np.array(xxs+xxf), delay=np.array(ds+df), z=toMatrix(zs+zf), p=toMatrix(ps+pf), 
        labels=(filters+windows),
        limits=[0.05, 80, -50, 10],
        plant=[wp, ws, gpass, gstop],
        colors=colors)


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
