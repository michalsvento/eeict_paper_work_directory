# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:54:43 2023

@author: Michal
"""

from mayavoz.models import Mayamodel
model = Mayamodel.from_pretrained("shahules786/mayavoz-waveunet-valentini-28spk")



import torch
from librosa import load
my_voice,sr = load("input_damaged.wav",sr=16000)
my_voice = torch.from_numpy(my_voice).unsqueeze(0).unsqueeze(0)


# %% 

#sf.write("out_denoise.wav",recon.numpy(),16000)

# audio = model.enhance(my,save_output=False)


audio = model.enhance(my_voice,sampling_rate=sr)
audio = audio.squeeze()
# print(audio.shape)

import matplotlib.pyplot as plt
fig2, [ax1,ax2] = plt.subplots(1, 2)

ax1.plot(audio)
ax2.plot(my_voice)

import soundfile as sf

sf.write('output.wav',audio,sr)

#audio = model.enhance(my_voice,save_output=True, sampling_rate=sr)

#Audio("cleaned_male.wav",rate=SAMPLING_RATE)

