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
        spectro = torch.stft(signal, self.n_fft,self.a,self.w,self.hann,return_complex=True)
        return spectro
    def idgt(self,spectro):
        signal = torch.istft(spectro, self.n_fft,self.a,self.w,torch.hann_window(self.w))
        return signal
    
    
tfa = Transform_audio(dgt_params)


# %% DRA functions and setup

def l1norm(signal):
    l1norm_signal = torch.linalg.norm(signal,ord=1)
    return l1norm_signal

def soft_thresh(signal,gamma):
    output = torch.sgn(signal)*torch.maximum((torch.abs(signal))-gamma,torch.tensor([0]))
    return output

def projection_from_spectrum(coeff,reference,gap):
    synthetized = tfa.idgt(coeff)  # Dz
    projn = synthetized.clone()
    projn[0:gap[0]]=reference[0:gap[0]]   # proj(Dz)
    projn[gap[1]:]=reference[gap[1]:]
    subtract = synthetized-projn #  Dz-proj(Dz)
    output = coeff - tfa.dgt(subtract)
    return output

def projection_time_domain(signal, reference, gap):
    signal_proj = signal.clone()
    signal_proj[0:gap[0]]=reference[0:gap[0]]     # proj(Dz)
    signal_proj[gap[1]:]=reference[gap[1]:]
    return signal_proj
    
dra_par ={
    "n_ite":1000,
    "lambda":1,
    "gamma":0.05
}


# %% DL model

class model(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(model, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out


denoise = model(1)


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

# %% DR algorithm 


# TODO: Signal loading  
# Test tensor

amplitude = np.iinfo(np.int16).max

Fs, orig_signal = wavfile.read("male.wav")
orig_signal = orig_signal[0:Fs*2]
pad_size = dgt_params['n_fft'] - len(orig_signal) % dgt_params['n_fft']
signal = np.concatenate((orig_signal,np.zeros([pad_size]))).astype(np.float64)
gap = np.array([Fs,Fs+800])  # pozicia diery
signal_norm = signal / amplitude
signal_t = torch.tensor(signal_norm) #.unsqueeze(0)
ref = signal_t.clone()
signal_t[gap[0]:gap[1]] = torch.tensor([0])




c = tfa.dgt(signal_t)
c_prev = c.clone()

normc = np.zeros([dra_par["n_ite"],1])
iterations = np.arange(1,dra_par["n_ite"]+1)
relative_change = np.zeros([dra_par["n_ite"]-1,1])

for i in tqdm(range(dra_par["n_ite"])):
    ci = projection_from_spectrum(c, ref, gap)
    if i>0:
        relative_change[i-1]=relative_sol_change(ci.flatten(), c_prev.flatten())
    c_prev = ci.clone()
    c = c+ dra_par["lambda"]*(soft_thresh(2*ci - c, dra_par["gamma"])-ci)  #Denoiser -> soft_thresh
    normc[i] = l1norm(c.squeeze(0))
 
 
#final_c = denoise.forward(c)
#reconstructed = tfa.idgt(final_c)


final_c = projection_from_spectrum(c, ref, gap)
reconstructed = tfa.idgt(final_c)

recon = reconstructed.numpy()
#scaled = np.int16(recon / np.max(np.abs(recon)) * 32767) + 32767/2
#wavfile.write('recon.wav', Fs,scaled)
    
# %% Save audio



sf.write('output.wav',recon,Fs)


# %% Metrics

# Short-Time Objective Intelligibility (STOI) 
stoi = ShortTimeObjectiveIntelligibility(Fs, False)
audio_stoi = stoi(reconstructed,ref)
print("STOI - full signal: ",audio_stoi.item())

# keď dopadne rekonštrukcia zle podobnosť je minimálna, STOI hlási error
# ze sa signaly vobec nepodobaju
# gap_stoi = stoi(reconstructed[gap[0]:gap[1]],ref[gap[0]:gap[1]])
# print("STOI - only gap: ",gap_stoi.item())

# Perceptual Evaluation of Speech Quality (PESQ)
pesq = PerceptualEvaluationSpeechQuality(Fs, 'nb')
pesq_val = pesq(reconstructed,ref)
print('PESQ (-0.5 - 4.5):',pesq_val.item())

# SNR - full length
snr_val = SNR(reconstructed,ref)
print("SNR (dB) - full length: ",snr_val.item())

# SNR - in gap
snr_val_gap = SNR(reconstructed[gap[0]:gap[1]],ref[gap[0]:gap[1]])
print("SNR (dB) - only gap: ",snr_val_gap.item())



# %% Visuals

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(11,3)


ax1.plot(iterations,normc)
ax1.set_title('l1norm over iterations');
ax1.set_xlabel('iterations')
ax1.set_ylabel('l1 norm')

fig2, [ax1,ax2] = plt.subplots(1, 2)

ax1.plot(recon[Fs-100:Fs+4000])
ax2.plot(signal_norm[Fs-100:Fs+4000])

fig3, ax1 = plt.subplots(1,1)

ax1.plot(iterations[:-1],relative_change)

