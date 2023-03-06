# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:27:29 2023

@author: Michal
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

start_time = datetime.date.today()#.strftime("%d/%m/%Y %H:%M:%S")

dgt_base = 1024

dgt_params = {
    "w": dgt_base,
    "a": int(dgt_base / 4),
    "n_fft": dgt_base
}



dra_par = {
    "n_ite": 1000,
    "lambda": 1,
    "gamma": 1.0
}

session_time = time.time()


df = pd.DataFrame({
    "session_date":start_time,
    "session_time":session_time,
    "n_ite":dra_par["n_ite"],
    "lambda":dra_par["lambda"],
    "gamma":dra_par["gamma"],
    "w":dgt_params["w"],
    "a":dgt_params['a'],
    "n_fft":dgt_params['n_fft'],
    "stoi":audio_stoi,
    "pesq":pesq_val,
    "snr":snr_val,
    "snr_gap",snr_val_gap
        },index=[1])