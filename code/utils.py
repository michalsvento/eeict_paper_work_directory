# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:50:04 2023

@author: Michal
"""

import torch



class Transform_audio():
    def __init__(self, param):
        self.w = param['w']
        self.a = param['a']
        self.n_fft = param['n_fft']
        self.hann = torch.hann_window(self.w, dtype=torch.float64)

    def dgt(self, signal):
        spectro = torch.stft(signal, self.n_fft, self.a, self.w, self.hann, return_complex=True, normalized=True, onesided=False)
        return spectro

    def idgt(self, spectro):
        signal = torch.istft(spectro, self.n_fft, self.a, self.w, self.hann, normalized=True,return_complex=True, onesided=False) #
        return signal
    
# %% Metrics functions

def SNR(reconstructed, reference):
    subtract_recon = reference - reconstructed
    l2_sub = torch.linalg.norm(subtract_recon, ord=2)
    l2_ref = torch.linalg.norm(reconstructed, ord=2)
    snr_ratio = 20 * torch.log10(l2_ref / l2_sub)
    return snr_ratio


def relative_sol_change(actual_sol, prev_sol):
    l2_actual = torch.linalg.norm(actual_sol - prev_sol, 2)
    l2_prev = torch.linalg.norm(prev_sol, 2)
    rel_change = l2_actual / l2_prev
    return rel_change


# %% DRA functions and setup

def l1norm(signal):
    l1norm_signal = torch.linalg.norm(signal, ord=1)
    return l1norm_signal


def soft_thresh(signal, gamma):
    output = torch.sgn(signal) * torch.maximum((torch.abs(signal)) - gamma, torch.tensor([0]))
    return output


def projection_time_domain(signal, reference, mask):
    signal_proj = signal.clone()
    signal_proj[mask] = reference[mask]
    return 0.8*signal_proj + 0.2*signal


def projection_time_domain_mix(signal, reference, mask, proj_percent):
    signal_proj = signal.clone()
    signal_proj[mask] = reference[mask]
    return (proj_percent)*signal_proj + (1-proj_percent)*signal

