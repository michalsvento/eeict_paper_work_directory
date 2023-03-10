# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:33:58 2023

@author: Michal
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import soundfile as sf
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import time
import pandas as pd
import os
import librosa
import datetime
from utils import Transform_audio,relative_sol_change,projection_time_domain,l1norm, SNR, soft_thresh


# %% SETUP


dgt_base = 1024

dgt_params = {
    "w": dgt_base,
    "a": int(dgt_base / 4),
    "n_fft": dgt_base
}


tfa = Transform_audio(dgt_params)

# %% DRA setup

dra_par = {
    "n_ite": 50,
    "lambda": 1,
    "lambda_n": 1,
    "gamma": 0.10,
    "eta": 0.5
}


session_date = datetime.date.today()

dir_name_vis = 'lambda_test'
print('Folder exists')if os.path.isdir(dir_name_vis) else os.mkdir(dir_name_vis)


# %% DL model

from mayavoz.models import Mayamodel
model = Mayamodel.from_pretrained("shahules786/mayavoz-waveunet-valentini-28spk")

# %% Signal loading  


orig_signal, Fs = librosa.load("noise119.wav",sr=16000)
ref_sig , Fs = librosa.load("clean119.wav",sr=16000)
orig_signal = orig_signal[0:Fs*2]
ref_sig = ref_sig[0:Fs*2]
pad_size = dgt_params['n_fft'] - len(orig_signal) % dgt_params['n_fft']
signal = np.concatenate((orig_signal,np.zeros([pad_size]))).astype(np.double())
ref_signal =  np.concatenate((ref_sig,np.zeros([pad_size]))).astype(np.double())


amplitude = np.iinfo(np.int16).max
signal_norm = signal #/ amplitude
signal_t = torch.from_numpy(signal_norm).float()#.unsqueeze(0)
signal_ref= torch.from_numpy(ref_signal).float()
ref = signal_ref.clone()


#gap = np.array([Fs,Fs+400])  # pozicia diery
threshold = 0.4  # i.e. 60% reliables
mask = np.random.default_rng(seed=42).uniform(0, 1, len(signal)) > threshold
mask = torch.tensor(mask)
# mask =  np.ones(signal.shape,dtype=bool)
# mask[round(0.5*Fs):round(0.5*Fs)+ round(Fs*0.03)] = 0
mask = torch.tensor(mask)
masksignal_t = signal_t.clone()
masksignal_t[~mask] = torch.tensor([0])
masksignal_t = masksignal_t  #0.05*torch.randn_like(masksignal_t)

signal_ID = 1

# %% Metrics

# Short-Time Objective Intelligibility (STOI) 
stoi = ShortTimeObjectiveIntelligibility(Fs, False)

# Perceptual Evaluation of Speech Quality (PESQ)
pesq = PerceptualEvaluationSpeechQuality(Fs, 'nb')

# lambda_table = np.logspace(-3,1,10,base=2.0,endpoint=False)  # lambda 0-1 
lambda_table = np.array([0.1,1])

# percentage divisions
# mix_fraction = np.arange(0,1.1,0.1)
snr_after_ite = np.zeros([dra_par["n_ite"],4])

for lam in range(len(lambda_table)):
    idx_ct = 2*lam
    print('ITE ',lam+1,'/',len(lambda_table))
    dra_par['lambda'] = lambda_table[lam]
    dra_par['lambda_n'] = lambda_table[lam]
    normx = np.zeros([dra_par["n_ite"],1])
    relative_change = np.zeros([dra_par["n_ite"]-1,2])
    iterations = np.arange(1,dra_par["n_ite"]+1)
    
    
    # %% DR algorithm orig
    
    print('CLASSIC PART')
    
    
    x = masksignal_t.clone()
    x_prev = x.clone()
    
    
    
    start_time = time.time()
    
    for i in tqdm(range(dra_par["n_ite"])):
        xi = projection_time_domain(x.float(), masksignal_t, mask)
        snr_after_ite[i,2*lam] = SNR(xi,ref)
        if i==1:
            xn = x.clone() 
        if i>0:
            relative_change[i-1,0]=relative_sol_change(xi, x_prev)
        x_prev = xi.clone()
        x =x + dra_par["lambda"]*(tfa.idgt(soft_thresh(tfa.dgt(2*xi - x), dra_par["gamma"]))-xi)  #Denoiser -> soft_thresh
        normx[i] = l1norm(tfa.dgt(x))
        dra_par["lambda"] =  dra_par["lambda"] * 0.9
    
    final_x = projection_time_domain(x.float(),ref,mask)
    
    SNR_max = np.max(snr_after_ite[:,0])
    SNR_idx = np.argmax(snr_after_ite[:,0])
    
    # %% Metrics
    
    
    audio_stoi = stoi(final_x,ref)
    print("STOI: ",audio_stoi.item())
    
    pesq_val = pesq(final_x,ref)
    print('PESQ (-0.5 - 4.5):',pesq_val.item())
    
    # SNR - full length
    snr_val = SNR(final_x,ref)
    print("SNR (dB) - full length: ",snr_val.item())
    
    # SNR - in gap
    snr_val_gap = SNR(final_x[~mask],ref[~mask])
    print("SNR (dB) - only gap: ",snr_val_gap.item())
    

    
    idx_ct = lam*2+1
     
        
    # %% Neural part
    
    print('NEURAL PART')
    start_time = time.time()
        
    xn = masksignal_t.clone()
    xn_prev = xn.clone()
    
    for i in tqdm(range(dra_par["n_ite"])):
        xin =  projection_time_domain(xn.float(), masksignal_t, mask)
        snr_after_ite[i,2*lam+1] = SNR(xn,ref)
        if i>0:
            relative_change[i-1,1]=relative_sol_change(xin, xn_prev)
        # if i>1 and i<dra_par["n_ite"]-1:
        #     if relative_change[i-1,1] > relative_change[i,1]: 
        #         dra_par["lambda"] = dra_par["lambda"]+0.1
        #     else:
        #         dra_par["lambda"] = dra_par["lambda"]-0.01
        # if (dra_par["lambda_n"]>=0 or dra_par["lambda_n"]<=2*dra_par["eta"]):
        #     dra_par["lambda_n"] =  dra_par["lambda_n"] * 0.9
        # else:
        #     dra_par["lambda_n"] = dra_par["eta"]
        xn_prev = xin.clone()
        xn =xn + dra_par["lambda_n"]*(tfa.idgt(soft_thresh(tfa.dgt(2*xin - xn), dra_par["gamma"]))-xin) #Denoiser -> soft_thresh
       
    final_xn = projection_time_domain(xn.float(),ref,mask)
    #final_xn = xn
    
    SNR_max = np.max(snr_after_ite[:,1])
    SNR_idx = np.argmax(snr_after_ite[:,1])
    
    audio_stoi = stoi(final_xn,ref)
    print("STOI: ",audio_stoi.item())
    
    pesq_val = pesq(final_xn.double(),ref)
    print('PESQ (-0.5 - 4.5):',pesq_val.item())
    
    # SNR - full length
    snr_val = SNR(final_xn,ref)
    print("SNR (dB) - full length: ",snr_val.item())
    
    # SNR - in gap
    snr_val_gap = SNR(final_xn[~mask],ref[~mask])
    print("SNR (dB) - only gap: ",snr_val_gap.item())
    
    
    end_time = time.time()
    session_time = end_time - start_time

    
    
    

    
    
    # %% Visuals

    
    now_str = datetime.datetime.strftime(datetime.datetime.now(),'%d-%m-%Y_%H-%M-%S')
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(11, 4)
    fig.suptitle(str(dra_par["n_ite"])+"iterations lambda:"+'{:.4e}'.format(dra_par["lambda"]))
    ax1.plot(iterations, snr_after_ite)
    ax1.set_title('l1norm over iterations');
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('l1 norm')

    






