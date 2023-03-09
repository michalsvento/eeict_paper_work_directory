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
import datetime
import time
import pandas as pd
import sys
import os
from utils import *
start_time = time.time()
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
    "w": dgt_base,
    "a": int(dgt_base / 4),
    "n_fft": dgt_base
}


# class Transform_audio():
#     def __init__(self, param):
#         self.w = param['w']
#         self.a = param['a']
#         self.n_fft = param['n_fft']
#         self.hann = torch.hann_window(self.w, dtype=torch.float64)

#     def dgt(self, signal):
#         spectro = torch.stft(signal, self.n_fft, self.a, self.w, self.hann, return_complex=True, normalized=True, onesided=False)
#         return spectro

#     def idgt(self, spectro):
#         signal = torch.istft(spectro, self.n_fft, self.a, self.w, self.hann, normalized=True, return_complex=True, onesided=False)
#         return signal


tfa = Transform_audio(dgt_params)


# %% DRA functions and setup

# def l1norm(signal):
#     l1norm_signal = torch.linalg.norm(signal, ord=1)
#     return l1norm_signal


# def soft_thresh(signal, gamma):
#     output = torch.sgn(signal) * torch.maximum((torch.abs(signal)) - gamma, torch.tensor([0]))
#     return output


# def projection_time_domain(signal, reference, mask):
#     signal_proj = signal.clone()
#     signal_proj[mask] = reference[mask]
#     return signal_proj


# def projection_from_spectrum(coeff, reference, mask):
#     synthetized = tfa.idgt(coeff)  # Dz
#     projn = projection_time_domain(synthetized, reference.type(synthetized.dtype), mask)
#     subtract = synthetized - projn  # Dz-proj(Dz)
#     output = coeff - tfa.dgt(subtract)
#     return output


dra_par = {
    "n_ite": 1000,
    "lambda": 1.5,
    "gamma": 1.0
}


# %% DL model

from mayavoz.models import Mayamodel
model = Mayamodel.from_pretrained("shahules786/mayavoz-waveunet-valentini-28spk")


# %% Metrics functions

# def SNR(reconstructed, reference):
#     subtract_recon = reference - reconstructed
#     l2_sub = torch.linalg.norm(subtract_recon, ord=2)
#     l2_ref = torch.linalg.norm(reconstructed, ord=2)
#     snr_ratio = 20 * torch.log10(l2_ref / l2_sub)
#     return snr_ratio


# def relative_sol_change(actual_sol, prev_sol):
#     l2_actual = torch.linalg.norm(actual_sol - prev_sol, 2)
#     l2_prev = torch.linalg.norm(prev_sol, 2)
#     rel_change = l2_actual / l2_prev
#     return rel_change


# %% DR algorithm


# Signal loading  

Fs, orig_signal = wavfile.read("male.wav")
orig_signal = orig_signal[0:Fs * 2]
pad_size = dgt_params['n_fft'] - len(orig_signal) % dgt_params['n_fft']
signal = np.concatenate((orig_signal, np.zeros([pad_size]))).astype(np.float64)

# Signal normalization and reference clone
amplitude = np.iinfo(np.int16).max
signal_norm = signal / amplitude
signal_t = torch.tensor(signal_norm)
ref = signal_t.clone()

threshold = 0.4  # i.e. 60% reliables
mu, sigma = 0.5, 0.5  # mean and standard deviation
mask = np.random.default_rng(seed=42).uniform(0, 1, len(signal)) > threshold
signal_t[~mask] = torch.tensor([0])

c = tfa.dgt(signal_t)
c_prev = c.clone()

normc = np.zeros([dra_par["n_ite"], 1])
iterations = np.arange(1, dra_par["n_ite"] + 1)
relative_change = np.zeros([dra_par["n_ite"] - 1, 1])

for i in tqdm(range(dra_par["n_ite"])):
    ci = projection_from_spectrum(c, ref, mask)
    if i > 0:
        relative_change[i - 1] = relative_sol_change(ci.flatten(), c_prev.flatten())
    c_prev = ci.clone()
    c = c + dra_par["lambda"] * (soft_thresh(2 * ci - c, dra_par["gamma"]) - ci)  # Denoiser -> soft_thresh
    # normc[i] = l1norm(c.squeeze(0))
    normc[i] = l1norm(ci.squeeze(0))

# final_c = denoise.forward(c)
# reconstructed = tfa.idgt(final_c)


final_c = projection_from_spectrum(c, ref, mask)
reconstructed = tfa.idgt(final_c)

print(f"norm of the imaginary part of signal solution: {torch.norm(torch.imag(reconstructed)):.3e}")

reconstructed = torch.real(reconstructed)

recon = reconstructed.numpy()

# %% Save audio


sf.write('output.wav', recon, Fs)
sf.write('input_damaged.wav', signal_t.numpy(), Fs)

# %% Metrics

# Short-Time Objective Intelligibility (STOI) 
stoi = ShortTimeObjectiveIntelligibility(Fs, False)
audio_stoi = stoi(reconstructed, ref)
print("STOI - full signal: ", audio_stoi.item())

# Perceptual Evaluation of Speech Quality (PESQ)
pesq = PerceptualEvaluationSpeechQuality(Fs, 'nb')
pesq_val = pesq(reconstructed, ref)
print('PESQ (-0.5 - 4.5):', pesq_val.item())

# SNR - full length
snr_val = SNR(reconstructed, ref)
print("SNR (dB) - full length: ", snr_val.item())

# SNR - in gap
snr_val_gap = SNR(reconstructed[~mask], ref[~mask])
print("SNR (dB) - only gap: ", snr_val_gap.item())

# %% Visuals

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(11, 3)

ax1.plot(iterations, normc)
ax1.set_title('l1norm over iterations');
ax1.set_xlabel('iterations')
ax1.set_ylabel('l1 norm')
plt.yscale("log")
fig.savefig("l1norm.png")

fig2, [ax1, ax2] = plt.subplots(1, 2)

ax1.plot(recon[Fs - 100:Fs + 4000])
ax2.plot(signal_norm[Fs - 100:Fs + 4000])
fig2.savefig("reconstruction.png")

fig3, ax1 = plt.subplots(1, 1)

ax1.plot(iterations[:-1], relative_change)
plt.yscale("log")
fig3.savefig("relative.png")



# %% Save data
# For further visuals


end_time = time.time()
session_time = end_time - start_time


df = pd.DataFrame({
    "session_date":start_time,
    "session_time":session_time,
    "n_ite":dra_par["n_ite"],
    "lambda":dra_par["lambda"],
    "gamma":dra_par["gamma"],
    "w":dgt_params["w"],
    "a":dgt_params['a'],
    "n_fft":dgt_params['n_fft'],
    "stoi":audio_stoi.numpy(),
    "pesq":pesq_val.numpy(),
    "snr":snr_val.numpy(),
    "snr_gap": snr_val_gap.numpy()
        },index=[1])



file = str(os.path.basename(__file__))
file = file[:-3]
output_name = file + '_'+str(dra_par["n_ite"])+"ite_lambda_"+'{:.2e}'.format(dra_par["lambda"])+".parquet"
dir_name = file+'_log'
if os.path.isdir(dir_name):
    df.to_parquet(dir_name+'/'+output_name)
else:
    os.mkdir(dir_name)
    df.to_parquet(dir_name+'/'+output_name)


if os.path.isfile('dra_speech_total.parquet'):
    df_tot = pd.read_parquet('dra_speech_total.parquet')
    df_tot = pd.concat([df_tot,df])
    df_tot.to_parquet('dra_speech_total.parquet')
else:
    df.to_parquet('dra_speech_total.parquet')
    
