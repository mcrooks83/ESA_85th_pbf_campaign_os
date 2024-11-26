from __future__ import division
import pandas as pd
import numpy as np
from scipy.signal import medfilt, butter, filtfilt, lfilter, sosfiltfilt, find_peaks, find_peaks_cwt,resample, detrend
import logging
import math
import time
import statistics as stats
import json
from datetime import datetime, timedelta
import os, sys
import glob

def read_in_clean_taltech_data(base):
    print(f"reading taltech data files from {base}")
    sensor = "taltech"
    data = {} # holds all the data
    ## read all files
    ## read all files
    env_dirs = [entry.name for entry in os.scandir(base) if entry.is_dir()]
    print(env_dirs)
    
    for env in env_dirs:
        # gets exp1 and exp2
        exp_dirs = [entry.name for entry in os.scandir(f"{base}/{env}") if entry.is_dir()]
        print(exp_dirs)
        # add an entry for the environment (baseline, flight)
        data[env] = {}
        for e in exp_dirs:
            data[env][e] = {} # add an entry the experimenter
            #path for an experimenter 
            exp_path = f"{base}/{env}/{e}/{sensor}"
            # get the shoes / barefoot dirs
            fw_dirs = [entry.name for entry in os.scandir(exp_path) if entry.is_dir()]
            print(fw_dirs)
            for f in fw_dirs:
                data[env][e][f] = {} # add an entry for the footwear
                fw_path = f"{exp_path}/{f}/*.csv"
    
                for file_path in glob.glob(fw_path):
                    f_name = file_path.split("\\")[1].split(".")[0]
                    data[env][e][f][f_name] = pd.read_csv(file_path)
    return data

def vector_magnitude(vectors):
    n = len(vectors[0])
    assert all(len(v) == n for v in vectors), "Vectors have different lengths"
    vm = np.sqrt(sum(v ** 2 for v in vectors))
    return vm

def build_filter(frequency, sample_rate, filter_type, filter_order):
    #nyq = 0.5 * sample_rate
    if filter_type == "band":
        #nyq_cutoff = (frequency[0] / nyq, frequency[1] / nyq)
        b, a = butter(filter_order, (frequency[0], frequency[1]), btype=filter_type, analog=False, output='ba', fs=sample_rate)
    elif filter_type == "low":
        #nyq_cutoff = frequency[1] / nyq
        b, a = butter(filter_order, frequency[1], btype=filter_type, analog=False, output='ba', fs=sample_rate)
    elif filter_type == "high":
        #nyq_cutoff = frequency[0] / nyq
        b, a = butter(filter_order, frequency[0], btype=filter_type, analog=False, output='ba', fs=sample_rate)

    return b, a


def filter_signal(b, a, signal, filter):
    if(filter=="lfilter"):
        return lfilter(b, a, signal)
    elif(filter=="filtfilt"):
        return filtfilt(b, a, signal)
    elif(filter=="sos"):
        return sosfiltfilt(sos, signal)
    

def compute_fft_mag(data):
    fftpoints = int(math.pow(2, math.ceil(math.log2(len(data)))))
    fft = np.fft.fft(data, n=fftpoints)
    mag = np.abs(fft) / (fftpoints/2) # check this
    return mag.tolist()

def fft_graph_values(fft_mags, sample_rate):
    T = 1/sample_rate
    N_r =len(fft_mags)//2
    x = np.linspace(0.0, 1.0/(2.0*T), len(fft_mags)//2).tolist()
    y = fft_mags[:N_r]
    
    return [x,y]

def compute_frequency_response(df, sampling_rate, b,a ):

    # data is a df
    a_x = df["Acc_X"]
    a_y = df["Acc_Y"]
    a_z = df["Acc_Z"]

    a_x = a_x.to_numpy()  / 9.80665
    a_y = a_y.to_numpy()  / 9.80665
    a_z = a_z.to_numpy()  / 9.80665

    a_mag = vector_magnitude([a_x, a_y, a_z])
    filtered_mag = filter_signal(b,a, a_mag, "filtfilt")
    
    fft_mag = compute_fft_mag(filtered_mag)
    graph = fft_graph_values(fft_mag, sampling_rate)
    return graph

# axis is a string either "Acc_X" "Acc_Y" "Acc_Z"
def compute_frequency_response_of_axis(df, axis, sampling_rate, b,a):

    # data is a df
    a_axis = df[axis]

    a_axis = a_axis.to_numpy()  / 9.80665
    
    filtered_a = filter_signal(b,a, a_axis, "filtfilt")
    
    fft_mag = compute_fft_mag(filtered_a)
    graph = fft_graph_values(fft_mag, sampling_rate)
    return graph

def compute_power_spectrum(fft_mag):
    power = np.square(fft_mag)
    return power

# computes the actual loading value
def compute_loading_intensity(fft_magnitudes, sampling_frequency, high_cut_off):
    fftpoints = int(math.pow(2, math.ceil(math.log2(len(fft_magnitudes)))))
    LI = 0
    fs = sampling_frequency
    fc = high_cut_off
    kc = int((fftpoints/fs)* fc) + 1

    magnitudes = fft_magnitudes

    f = []
    for i in range(0, int(fftpoints/2)+1):
        f.append((fs*i)/fftpoints)

    for k in range(0, kc):
        LI = LI + (magnitudes[k] * f[k])

    return LI


# computes the total skeletal loading (assumes df columns as input)

def compute_skeletal_loading(accel_x, accel_y, accel_z, sampling_rate, lc_off, hc_off, filter_order, filter_type):
    # build the filter
    b,a = build_filter((lc_off, hc_off), sampling_rate, filter_type, filter_order)
    
    accel_x = accel_x.to_numpy()  
    accel_y = accel_y.to_numpy()  
    accel_z = accel_z.to_numpy()  
    
    # compute the magnitude vector
    a_mag = vector_magnitude([accel_x, accel_y, accel_z])
    # filter the magnitude
    filtered_mag = filter_signal(b,a, a_mag, "filtfilt")
    # compute the frequency response
    fft_mag = compute_fft_mag(filtered_mag)
    #fft_graph = compute_frequency_response(df, sampling_rate, b,a )
    # compute the loading intensity
    li_result = compute_loading_intensity(fft_mag, sampling_rate, hc_off)
        
    return li_result

# computes the total skeletal loading for a given axis
# assumes a df column input
# this could compute the total loading if the magnitude is passed in as the axis
def compute_skeletal_loading_axis(axis,sampling_rate, lc_off, hc_off, filter_order, filter_type):
    # build the filter
    b,a = build_filter((lc_off, hc_off), sampling_rate, filter_type, filter_order)
    
    axis = axis.to_numpy()

    filtered_mag = filter_signal(b,a, axis, "filtfilt")
    fft_mag = compute_fft_mag(filtered_mag)
    li_result = compute_loading_intensity(fft_mag, sampling_rate, hc_off)
        
    return li_result


# computes all the metrics for a df
def compute_skeletal_loading_metrics(df, sampling_rate, lc_off, hc_off, filter_order, filter_type):
    # extract the axes
    a_x = df["X"]
    a_y = df["Y"]
    a_z = df["Z"]

    # total loading
    total_li = compute_skeletal_loading(a_x, a_y, a_z, 
                                        sampling_rate, 
                                        lc_off, 
                                        hc_off, 
                                        filter_order, filter_type)

    # axis loading
    x_li = compute_skeletal_loading_axis(a_x,
                                         sampling_rate, 
                                         lc_off, hc_off, 
                                         filter_order, filter_type)
    y_li = compute_skeletal_loading_axis(a_y,
                                         sampling_rate, 
                                         lc_off, hc_off, 
                                         filter_order, filter_type)
    z_li = compute_skeletal_loading_axis(a_z,
                                         sampling_rate, 
                                         lc_off, hc_off, 
                                         filter_order, filter_type)

    return round(total_li,2), round(x_li,2), round(y_li,2), round(z_li,2)




