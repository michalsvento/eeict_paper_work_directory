# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:34:09 2023

@author: Michal
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.io import wavfile
import soundfile as sf
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

# %% TFAnalysis setup

# torch.stft is used - more info in pytorch documentation
# =============================================================================
# # Discrete Gabor transform 
# # Window - ('gauss','hann', 'hanning')
#   w - window_length
#   a - hop length,best fraction of w, egg. w/2, w/4,
#   nfft -length of fast fourier transform
# =============================================================================

dgt_base = 1024

dgt_params = {
    "w":dgt_base,
    "a":int(dgt_base/4),
    "n_fft":dgt_base
}



class Transform_audio():
    def __init__(self,param):
        self.w = param['w']
        self.a = param['a']
        self.n_fft = param['n_fft']
        self.hann = torch.hann_window(self.w)
        
    def dgt(self,signal):
        spectro = torch.stft(signal, self.n_fft,self.a,self.w,self.hann,return_complex=True,normalized=True)
        return spectro
    def idgt(self,spectro):
        signal = torch.istft(spectro, self.n_fft,self.a,self.w,self.hann,normalized=True)
        return signal
    
    
tfa = Transform_audio(dgt_params)


# %% DRA functions and setup

def l1norm(signal):
    l1norm_signal = torch.linalg.norm(signal,ord=1)
    return l1norm_signal

def soft_thresh(signal,gamma):
    output = torch.sgn(signal)*torch.maximum((torch.abs(signal))-gamma,torch.tensor([0]))
    return output

def projection_time_domain(signal, reference, mask):
    signal_proj = signal.clone()
    signal_proj[mask] = reference[mask]
    return signal_proj
    
dra_par ={
    "n_ite":1000,
    "lambda":1,
    "gamma":0.1,
    "alfa":0.5
}


# %% Metrics functions

def SNR(reconstructed, reference):
    subtract_recon = reference-reconstructed
    l2_sub = torch.linalg.norm(subtract_recon,ord=2)
    l2_ref = torch.linalg.norm(reconstructed,ord=2)
    snr_ratio = 20*torch.log10(l2_ref/l2_sub)
    return snr_ratio

def relative_sol_change(actual_sol, prev_sol):
    l2_actual = torch.linalg.norm(actual_sol-prev_sol,2)
    l2_prev = torch.linalg.norm(prev_sol,2)
    rel_change = l2_actual/l2_prev
    return rel_change


# %% Signal loading  


Fs, orig_signal = wavfile.read("male.wav")
orig_signal = orig_signal[0:Fs*2]
pad_size = dgt_params['n_fft'] - len(orig_signal) % dgt_params['n_fft']
signal = np.concatenate((orig_signal,np.zeros([pad_size]))).astype(np.float64)

amplitude = np.iinfo(np.int16).max
signal_norm = signal / amplitude
signal_t = torch.tensor(signal_norm) #.unsqueeze(0)
ref = signal_t.clone()


#gap = np.array([Fs,Fs+400])  # pozicia diery
threshold = 0.4  # i.e. 60% reliables
mu, sigma = 0.5, 0.5 # mean and standard deviation
mask = np.random.default_rng(seed=42).normal(mu,sigma,len(signal))> threshold
signal_t[~mask] = torch.tensor([0])


# %% DR algorithm

#c = tfa.dgt(signal_t)

x = signal_t.clone()
x_prev = x.clone()

normx = np.zeros([dra_par["n_ite"],1])
relative_change = np.zeros([dra_par["n_ite"]-1,1])
iterations = np.arange(1,dra_par["n_ite"]+1)

for i in tqdm(range(dra_par["n_ite"])):
    xi = projection_time_domain(x, ref, mask)
    if i>0:
        relative_change[i-1]=relative_sol_change(xi, x_prev)
    x_prev = xi.clone()
    x =x + dra_par["lambda"]*(tfa.idgt(soft_thresh(tfa.dgt(2*xi - x), dra_par["gamma"]))-xi)  #Denoiser -> soft_thresh
    normx[i] = l1norm(tfa.dgt(x))


final_x = projection_time_domain(x,ref,mask)
    
# %% Save audio

sf.write('output_td.wav',final_x,Fs)


# %% Metrics

# Short-Time Objective Intelligibility (STOI) 
stoi = ShortTimeObjectiveIntelligibility(Fs, False)
audio_stoi = stoi(final_x,ref)
print("STOI - full signal: ",audio_stoi.item())

# Perceptual Evaluation of Speech Quality (PESQ)
pesq = PerceptualEvaluationSpeechQuality(Fs, 'nb')
pesq_val = pesq(final_x,ref)
print('PESQ (-0.5 - 4.5):',pesq_val.item())

# SNR - full length
snr_val = SNR(final_x,ref)
print("SNR (dB) - full length: ",snr_val.item())

# SNR - in gap
snr_val_gap = SNR(final_x[~mask],ref[~mask])
print("SNR (dB) - only gap: ",snr_val_gap.item())






# %% Visuals

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(11,3)


ax1.plot(iterations,normx)
ax1.set_title('l1norm over iterations');
ax1.set_xlabel('iterations')
ax1.set_ylabel('l1 norm')

fig2, [ax1,ax2] = plt.subplots(1, 2)

# ax1.plot(final_x[Fs-100:Fs+4000])
# ax2.plot(signal_norm[Fs-100:Fs+4000])

ax1.plot(final_x)
ax2.plot(signal_norm)

fig3, ax1 = plt.subplots(1, 1)
ax1.plot(iterations[:-1],relative_change)
ax1.set_title('l2norm relative change');
