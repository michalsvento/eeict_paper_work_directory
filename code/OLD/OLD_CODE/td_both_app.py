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
    "gamma": 1.0
}


session_date = datetime.date.today()

dir_name_vis = 'td_1'
print('Folder exists')if os.path.isdir(dir_name_vis) else os.mkdir(dir_name_vis)


    
    
d_l1 = '11_norm'
d_l2 = 'l2_relative'
d_sol = 'waveforms_png'
d_snr = 'snr'
d_wave = 'wavs'

print('Subfolder' + d_l1+' exists')if os.path.isdir(dir_name_vis+'/'+d_l1) else os.mkdir(dir_name_vis+'/'+d_l1)
print('Subfolder '+ d_l2+'exists')if os.path.isdir(dir_name_vis+'/'+d_l2) else os.mkdir(dir_name_vis+'/'+d_l2)
print('Subfolder'+d_sol+' exists')if os.path.isdir(dir_name_vis+'/'+d_sol) else os.mkdir(dir_name_vis+'/'+d_sol)
print('Subfolder'+d_wave+' exists')if os.path.isdir(dir_name_vis+'/'+d_wave) else os.mkdir(dir_name_vis+'/'+d_wave)
print('Subfolder'+d_snr+' exists')if os.path.isdir(dir_name_vis+'/'+d_snr) else os.mkdir(dir_name_vis+'/'+d_snr)

# %% DL model

from mayavoz.models import Mayamodel
model = Mayamodel.from_pretrained("shahules786/mayavoz-waveunet-valentini-28spk")

# %% Signal loading  


orig_signal, Fs = librosa.load("male.wav",sr=None)
orig_signal = orig_signal[0:Fs*2]
pad_size = dgt_params['n_fft'] - len(orig_signal) % dgt_params['n_fft']
signal = np.concatenate((orig_signal,np.zeros([pad_size]))).astype(np.double())

amplitude = np.iinfo(np.int16).max
signal_norm = signal #/ amplitude
signal_t = torch.from_numpy(signal_norm).float()#.unsqueeze(0)
ref = signal_t.clone()


#gap = np.array([Fs,Fs+400])  # pozicia diery
threshold = 0.4  # i.e. 60% reliables
mask = np.random.default_rng(seed=42).uniform(0, 1, len(signal)) > threshold
mask = torch.tensor(mask)
signal_t[~mask] = torch.tensor([0])

signal_ID = 1

# %% Metrics

# Short-Time Objective Intelligibility (STOI) 
stoi = ShortTimeObjectiveIntelligibility(Fs, False)

# Perceptual Evaluation of Speech Quality (PESQ)
pesq = PerceptualEvaluationSpeechQuality(Fs, 'nb')

lambda_table = np.logspace(-3,1,15,base=2.0)

for lam in range(len(lambda_table)):
    idx_ct = lam*2+1
    print('LAMBDA ',lam+1,'/',len(lambda_table))
    dra_par['lambda'] = lambda_table[lam]
    dra_par['lambda_n'] = lambda_table[lam]
    
    normx = np.zeros([dra_par["n_ite"],1])
    relative_change = np.zeros([dra_par["n_ite"]-1,2])
    iterations = np.arange(1,dra_par["n_ite"]+1)
    snr_after_ite = np.zeros([dra_par["n_ite"],2])
    
    # %% DR algorithm orig
    
    print('CLASSIC PART')
    
    
    x = signal_t.clone()
    x_prev = x.clone()
    
    
    
    start_time = time.time()
    
    for i in tqdm(range(dra_par["n_ite"])):
        xi = projection_time_domain(x.float(), ref, mask)
        snr_after_ite[i,0] = SNR(xi,ref)
        if i>0:
            relative_change[i-1,0]=relative_sol_change(xi, x_prev)
        x_prev = xi.clone()
        x =x + dra_par["lambda"]*(tfa.idgt(soft_thresh(tfa.dgt(2*xi - x), dra_par["gamma"]))-xi)  #Denoiser -> soft_thresh
        normx[i] = l1norm(tfa.dgt(x))
    
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
    
    # %% Save data
    
    # For further visuals
    
    
    end_time = time.time()
    session_time = end_time - start_time
    
    
    df = pd.DataFrame({
        "signal_ID":signal_ID,
        "approach":"conventional",
        "session_date":session_date,
        "session_time":session_time,
        "n_ite":dra_par["n_ite"],
        "lambda":dra_par["lambda"],
        "gamma":dra_par["gamma"],
        "w":dgt_params["w"],
        "a":dgt_params['a'],
        "n_fft":dgt_params['n_fft'],
        "stoi":audio_stoi.item(),
        "pesq":pesq_val.item(),
        "snr":snr_val.item(),
        "snr_gap": snr_val_gap.item(),
        "SNR_MAX": SNR_max,
        "SNR_IDX": SNR_idx,
        "rel_change": relative_change[-1,0]
            },index=[idx_ct])
    
    idx_ct = lam*2
    
    file = 'td'
    output_name = file + '_'+str(dra_par["n_ite"])+"ite_lambda_"+'{:.2e}'.format(dra_par["lambda"])+".parquet"
    dir_name = file+'_log'
    if os.path.isdir(dir_name_vis+'/'+dir_name):
        df.to_parquet(dir_name_vis+'/'+dir_name+'/'+output_name)
    else:
        os.mkdir(dir_name_vis+'/'+dir_name)
        df.to_parquet(dir_name_vis+'/'+dir_name+'/'+output_name)
    
    if os.path.isfile(dir_name_vis+'/dra_speech_total.parquet'):
        df_tot = pd.read_parquet(dir_name_vis+'/dra_speech_total.parquet')
        df_tot = pd.concat([df_tot,df])
        df_tot.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
    else:
        df.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
        
    # %% Neural part
    
    print('NEURAL PART')
    start_time = time.time()
        
    xn = signal_t.clone()
    xn_prev = xn.clone()
    
    for i in tqdm(range(dra_par["n_ite"])):
        xin = projection_time_domain(xn.float(), ref, mask)
        snr_after_ite[i,1] = SNR(xin,ref)
        if i>0:
            relative_change[i-1,1]=relative_sol_change(xin, xn_prev)
        # if i>1 and i<dra_par["n_ite"]-1:
        #     if relative_change[i-1,1] > relative_change[i,1]: 
        #         dra_par["lambda"] = dra_par["lambda"]+0.1
        #     else:
        #         dra_par["lambda"] = dra_par["lambda"]-0.01
        dra_par["lambda_n"] = 0.001 if dra_par["lambda_n"]<=0 else dra_par["lambda_n"]-0.01
        xn_prev = xin.clone()
        xn =xn + dra_par["lambda"]*(model.enhance(2*xin - xn,sampling_rate=Fs).squeeze()-xin) #Denoiser -> soft_thresh
       
    final_xn = projection_time_domain(xn.float(),ref,mask)
    
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
    
    df = pd.DataFrame({
        "signal_ID":signal_ID,
        "approach":'Denoiser',
        "session_date":session_date,
        "session_time":session_time,
        "n_ite":dra_par["n_ite"],
        "lambda":dra_par["lambda"],
        "gamma":dra_par["gamma"],
        "w":dgt_params["w"],
        "a":dgt_params['a'],
        "n_fft":dgt_params['n_fft'],
        "stoi":audio_stoi.item(),
        "pesq":pesq_val.item(),
        "snr":snr_val.item(),
        "snr_gap": snr_val_gap.item(),
        "SNR_MAX": SNR_max,
        "SNR_IDX": SNR_idx,
        "rel_change": relative_change[-1,1]
        
            },index=[idx_ct])
    
    
    
    file = 'td'
    output_name = file + '_'+str(dra_par["n_ite"])+"ite_lambda_"+'{:.2e}'.format(dra_par["lambda"])+".parquet"
    dir_name = file+'_log'
    if os.path.isdir(dir_name_vis+'/'+dir_name):
        df.to_parquet(dir_name_vis+'/'+dir_name+'/'+output_name)
    else:
        os.mkdir(dir_name_vis+'/'+dir_name)
        df.to_parquet(dir_name_vis+'/'+dir_name+'/'+output_name)
    
    
    if os.path.isfile(dir_name_vis+'/dra_speech_total.parquet'):
        df_tot = pd.read_parquet(dir_name_vis+'/dra_speech_total.parquet')
        df_tot = pd.concat([df_tot,df])
        df_tot.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
    else:
        df.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
    
    
    # %% Visuals

    
    now_str = datetime.datetime.strftime(datetime.datetime.now(),'%d-%m-%Y_%H-%M-%S')
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(11, 4)
    fig.suptitle(str(dra_par["n_ite"])+"iterations lambda:"+'{:.4e}'.format(dra_par["lambda"]))
    ax1.plot(iterations, normx)
    ax1.set_title('l1norm over iterations');
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('l1 norm')
    plt.yscale("log")
    fig.savefig(dir_name_vis+'/'+d_l1+'/'+now_str+'_norm_l1.png')
    
    fig2, [ax1, ax2,ax3] = plt.subplots(1, 3)
    fig2.set_size_inches(11, 4)
    fig2.suptitle(str(dra_par["n_ite"])+"iterations lambda:"+'{:.4e}'.format(dra_par["lambda"]))
    ax1.plot(final_x)
    ax1.set_title('conventional')
    ax2.plot(signal_norm)
    ax2.set_title('reference')
    ax3.plot(final_xn)
    ax3.set_title('denoiser')
    fig2.savefig(dir_name_vis+'/'+d_sol+'/'+now_str+'_waveforms.png')
    
    fig3, ax1 = plt.subplots(1, 1)
    fig3.suptitle(str(dra_par["n_ite"])+"iterations lambda:"+'{:.4e}'.format(dra_par["lambda"]))
    fig3.set_size_inches(11, 4)
    ax1.plot(iterations[:-1], relative_change)
    ax1.set_title('Relative_change')
    ax1.legend(['conv','denoiser'])
    plt.yscale("log")
    fig3.savefig(dir_name_vis+'/'+d_l2+'/'+now_str+'_rel_change.png')
    
    fig4, ax1 = plt.subplots(1, 1)
    fig4.set_size_inches(11, 4)
    fig4.suptitle(str(dra_par["n_ite"])+"iterations lambda:"+'{:.4e}'.format(dra_par["lambda"]))
    ax1.plot(iterations, snr_after_ite)
    ax1.set_title('SNR')
    ax1.legend(['conv','denoiser'])
    plt.yscale("log")
    fig4.savefig(dir_name_vis+'/'+d_snr+'/'+now_str+'_snr.png')
    
    
    # %% Save audio
    
    sf.write(dir_name_vis+'/'+d_wave+'/'+now_str+'_classic.wav',final_x,Fs)
    sf.write(dir_name_vis+'/'+d_wave+'/'+now_str+'_neural.wav',final_xn,Fs)

