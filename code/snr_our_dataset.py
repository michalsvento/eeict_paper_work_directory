# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:05:48 2023

@author: Michal
"""


import numpy as np
import os
import librosa
import glob

clean_fol = 'CLEAN'
noise_fol = 'NOISE'
clean_files = glob.glob1(clean_fol,'*.wav')
noise_files = glob.glob1(noise_fol,'*.wav')


snr1 = np.zeros([10,1])

def SNR(reconstructed, reference):
    subtract_recon = reference - reconstructed 
    l2_sub = np.linalg.norm(subtract_recon, ord=2)
    l2_ref =np.linalg.norm(reconstructed, ord=2)
    snr_ratio = 20 * np.log10(l2_ref / l2_sub)
    return snr_ratio

for i in range(len(snr1)):
    clean, sr = librosa.load(clean_fol+'/'+clean_files[i],sr=None)
    noise, sr = librosa.load(noise_fol+'/'+noise_files[i],sr=None)
    snr1[i,0] = SNR(noise,clean)
    






