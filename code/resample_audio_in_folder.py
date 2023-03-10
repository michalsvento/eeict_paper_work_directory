# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:39:01 2023

@author: Michal
"""


import soundfile as sf
import glob
import librosa
import os


def resample_folder(input_folder='./',output_folder='./', target_sr=16000):
    files = glob.glob1(input_folder,'*.wav')
    print("Folder exist") if os.path.isdir(output_folder) else os.mkdir(output_folder)
    for n in range(len(files)):
        sound, sr = librosa.load(files[0])
        res_sound = librosa.resample(sound,orig_sr=sr,target_sr=target_sr)
        sf.write(files[n], res_sound,target_sr)

#folder= './'
#target_sr = 16000
resample_folder()
                
    
    
