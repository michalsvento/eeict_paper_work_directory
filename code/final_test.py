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
from utils import Transform_audio,relative_sol_change,projection_time_domain_mix,l1norm, SNR, soft_thresh
import glob

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

dir_name_vis = 'final_test_1'
print('Folder exists')if os.path.isdir(dir_name_vis) else os.mkdir(dir_name_vis)



# %% Preparation

clean_fol = 'CLEAN'
noise_fol = 'NOISE'
clean_files = glob.glob1(clean_fol,'*.wav')
noise_files = glob.glob1(noise_fol,'*.wav')



# %% DL model

from mayavoz.models import Mayamodel
model = Mayamodel.from_pretrained("shahules786/mayavoz-waveunet-valentini-28spk")

# %% Signal loading  


for no_sig in range(len(clean_files)):
    signal_ID = clean_files[no_sig]
    orig_signal, Fs = librosa.load(noise_fol +'/'+noise_files[no_sig],sr=16000)
    ref_sig , Fs = librosa.load(clean_fol +'/'+clean_files[no_sig],sr=16000)
    pad_size = dgt_params['n_fft'] - len(orig_signal) % dgt_params['n_fft']
    signal = np.concatenate((orig_signal,np.zeros([pad_size]))).astype(np.double())
    ref_signal =  np.concatenate((ref_sig,np.zeros([pad_size]))).astype(np.double())
    
    
    signal_norm = signal #/ amplitude
    signal_t = torch.from_numpy(signal_norm).float()#.unsqueeze(0)
    signal_ref= torch.from_numpy(ref_signal).float()
    ref = signal_ref.clone()
    
    
    #gap = np.array([Fs,Fs+400])  # pozicia diery
    threshold = 0.4  # i.e. 60% reliables
    mask = np.random.default_rng(seed=42).uniform(0, 1, len(signal)) > threshold
    mask = torch.tensor(mask)
    masksignal_t = signal_t.clone()
    masksignal_t[~mask] = torch.tensor([0])
    #masksignal_t = masksignal_t +0.05*torch.randn_like(masksignal_t)
    
    
    
    # %% Metrics
    
    # Short-Time Objective Intelligibility (STOI) 
    stoi = ShortTimeObjectiveIntelligibility(Fs, False)
    
    # Perceptual Evaluation of Speech Quality (PESQ)
    pesq = PerceptualEvaluationSpeechQuality(Fs, 'nb')
    
    # lambda_table = np.logspace(-3,1,10,base=2.0,endpoint=False)  # lambda 0-1 
    lambda_table = np.array([1])
    
    # percentage divisions
    mix_fraction = np.arange(0,1.1,0.1)
    
    for lam in range(len(mix_fraction)):
        idx_ct = 3*lam + no_sig*(3*len(mix_fraction))
        print('ITE ',lam+1,'/',len(mix_fraction))
        dra_par['lambda'] = 1#lambda_table[lam]
        dra_par['lambda_n'] = 1#lambda_table[lam]
        normx = np.zeros([dra_par["n_ite"],1])
        relative_change = np.zeros([dra_par["n_ite"]-1,3])
        iterations = np.arange(1,dra_par["n_ite"]+1)
        snr_after_ite = np.zeros([dra_par["n_ite"],3])
        
        # %% DR algorithm orig
        
        dra_par["n_ite"]=50
        
        print('CLASSIC PART')
        
        x = masksignal_t.clone()
        x_prev = x.clone()
        
        start_time = time.time()
        
        for i in tqdm(range(dra_par["n_ite"])):
            xi = projection_time_domain_mix(x.float(), masksignal_t, mask,mix_fraction[lam])
            snr_after_ite[i,0] = SNR(xi,ref)
            if i==1:
                xn = x.clone() 
            if i>0:
                relative_change[i-1,0]=relative_sol_change(xi, x_prev)
            x_prev = xi.clone()
            x =x + dra_par["lambda"]*(tfa.idgt(soft_thresh(tfa.dgt(2*xi - x), dra_par["gamma"]))-xi)  #Denoiser -> soft_thresh
            normx[i] = l1norm(tfa.dgt(x))
            dra_par["lambda"] =  dra_par["lambda"] * 0.9
        
        final_x = projection_time_domain_mix(x.float(),ref,mask,mix_fraction[lam])
        
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
            "rel_change": relative_change[-1,0],
            "proj_frac":mix_fraction[lam]
                },index=[idx_ct+1])
        

        
        if os.path.isfile(dir_name_vis+'/dra_speech_total.parquet'):
            df_tot = pd.read_parquet(dir_name_vis+'/dra_speech_total.parquet')
            df_tot = pd.concat([df_tot,df])
            df_tot.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
        else:
            df.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
            
        # %% Neural part
        
        print('NEURAL PART')
        start_time = time.time()
            
        xn = masksignal_t.clone()
        xn_prev = xn.clone()
        
        for i in tqdm(range(dra_par["n_ite"])):
            xin =  projection_time_domain_mix(xn.float(), masksignal_t, mask, mix_fraction[lam])
            snr_after_ite[i,1] = SNR(xn,ref)
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
            dra_par["lambda_n"] =  dra_par["lambda_n"] * 0.9
            xn_prev = xin.clone()
            xn =xn + dra_par["lambda_n"]*(model.enhance(2*xin - xn,sampling_rate=Fs).squeeze()-xin) #Denoiser -> soft_thresh
           
        final_xn = projection_time_domain_mix(xn,ref,mask,mix_fraction[lam])
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
            "rel_change": relative_change[-1,1],
            "proj_frac": mix_fraction[lam]
                },index=[idx_ct+1])
    
    
        
        if os.path.isfile(dir_name_vis+'/dra_speech_total.parquet'):
            df_tot = pd.read_parquet(dir_name_vis+'/dra_speech_total.parquet')
            df_tot = pd.concat([df_tot,df])
            df_tot.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
        else:
            df.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
    
    
    # %% DRA _ 500 ite
    # %% DR algorithm orig
        
        print('CLASSIC PART')
        
        x = masksignal_t.clone()
        x_prev = x.clone()
        
        start_time = time.time()
        
        dra_par["n_ite"]=500
        
        for i in tqdm(range(dra_par["n_ite"])):
            xi = projection_time_domain_mix(x.float(), masksignal_t, mask,mix_fraction[lam])
            snr_after_ite[i,0] = SNR(xi,ref)
            if i==1:
                xn = x.clone() 
            if i>0:
                relative_change[i-1,0]=relative_sol_change(xi, x_prev)
            x_prev = xi.clone()
            x =x + dra_par["lambda"]*(tfa.idgt(soft_thresh(tfa.dgt(2*xi - x), dra_par["gamma"]))-xi)  #Denoiser -> soft_thresh
            normx[i] = l1norm(tfa.dgt(x))
            dra_par["lambda"] =  dra_par["lambda"] * 0.9
        
        final_x = projection_time_domain_mix(x.float(),ref,mask,mix_fraction[lam])
        
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
            "rel_change": relative_change[-1,0],
            "proj_frac":mix_fraction[lam]
                },index=[idx_ct+2])
        
        
        
        if os.path.isfile(dir_name_vis+'/dra_speech_total.parquet'):
            df_tot = pd.read_parquet(dir_name_vis+'/dra_speech_total.parquet')
            df_tot = pd.concat([df_tot,df])
            df_tot.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
        else:
            df.to_parquet(dir_name_vis+'/dra_speech_total.parquet')
    




