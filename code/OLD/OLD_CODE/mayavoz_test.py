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
my_voice = torch.from_numpy(my_voice)  #.unsqueeze(0).unsqueeze(0)


# %% 

#sf.write("out_denoise.wav",recon.numpy(),16000)

# audio = model.enhance(my,save_output=False)



audio = model.enhance(my_voice,sampling_rate=sr)
#audio = audio.squeeze()
# print(audio.shape)

for x in range(5):
    audio = model.enhance(audio,sampling_rate=sr)

import matplotlib.pyplot as plt
fig2, [ax1,ax2] = plt.subplots(1, 2)

ax1.plot(audio.squeeze())
ax2.plot(my_voice)

import soundfile as sf

# sf.write('male_denoise2.wav',audio.squeeze(),sr)

#audio = model.enhance(my_voice,save_output=True, sampling_rate=sr)

#Audio("cleaned_male.wav",rate=SAMPLING_RATE)

# %% correl
# import torch.nn.functional as F
import numpy as np
out_np = audio.squeeze().numpy()
in_np = my_voice.numpy()
inout_xcorr = np.correlate(out_np,in_np,'full')

qq = np.arange(-len(out_np)+1,len(out_np)) # od - 32767 po 32767  M+N-1

fig, ax1 = plt.subplots(1, 1)




ax1.plot(qq,inout_xcorr)
ax1.set_xlim([-20,20])


idx_max = np.argmax(inout_xcorr)

print(qq[idx_max])