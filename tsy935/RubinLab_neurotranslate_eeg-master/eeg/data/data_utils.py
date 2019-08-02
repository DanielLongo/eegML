import numpy as np
import pandas as pd
import h5py
import random
import os
import sys
import inspect
import pyedflib
import time
import cv2
from tqdm import tqdm
from glob import glob
from scipy.fftpack import fft, diff
from scipy.signal import hilbert, convolve2d
from sklearn.model_selection import StratifiedKFold, train_test_split

# Add parent dir to the sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from constants import *

def getOrderedChannels(file_name, verbose, labels_object):
    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split('-')[0]

    ordered_channels = []
    for ch in INCLUDED_CHANNELS:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if (verbose): 
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def getSeizureTimes(file_name, file_type = "edf"):
    """
    Args:
        file_name: file name of .edf file etc.
        file_type: "edf" or "tse"
    Returns:
        seizure_times: list of times of seizure onset in seconds     
    """
    tse_file = ""
    if file_type == "edf":
        tse_file = file_name[:-4] + ".tse"
    elif file_type == "tse":
        tse_file = file_name
    else:
        raise valueError("Unrecognized file type.")
    
    seizure_times = []
    with open(tse_file) as f:
        for line in f.readlines():
            #if "sz" in line: # if seizure
            if any(s in line for s in TARGET_LABS): # if this is one of the seizure types of interest
                # seizure start and end time
                seizure_times.append([float(line.strip().split(' ')[0]), float(line.strip().split(' ')[1])])
    return seizure_times


def getSeizureClass(file_name, file_type = "edf"):
    """
    Args:
        file_name: file name of .edf file etc.
        file_type: "edf" or "tse"
    Returns:
        seizure_class: list of seizure class in the .edf file      
    """
    tse_file = ""
    if file_type == "edf":
        tse_file = file_name[:-4] + ".tse"
    elif file_type == "tse":
        tse_file = file_name
    else:
        raise valueError("Unrecognized file type.")
        
    seizure_class = []
    with open(tse_file) as f:
        for line in f.readlines():
            if any(s in line for s in TARGET_LABS): # if this is one of the seizure types of interest
                seizure_str = [s for s in TARGET_LABS if s in line]
                print('seizure str:{}'.format(seizure_str))
                seizure_class.append(LABEL_DICT[seizure_str[0]])
    return seizure_class


def getSamplingFreq(edf, ordered_channels):
    """
    Args:
        edf: EDF file object
        ordered_channels: list of indices of channels we're interested in, according to the order in INCLUDED_CHANNELS
    Returns:
        frequencies: list of sampling frequencies of the EEG signals in each channel
    """
    frequencies = []
    for ch in ordered_channels:
        frequencies.append(edf.getSampleFrequency(ch))
    
    return frequencies


def getEDFsignals(edf):
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i,:] = edf.readSignal(i)
        except:
            pass
    return signals


def shuffleByPatients(filepath):
    level_1 = os.listdir(filepath)
    patients = []
    for folder in level_1:
        print(folder)
        for people in os.listdir(os.path.join(filepath, folder)):
            patients.append(os.path.join(filepath, folder, people))
    random.shuffle(patients)
    res = []
    for people in patients:
        res.extend([y for x in os.walk(people) for y in glob(os.path.join(x[0], '*.edf'))])
    return res


# Write seizure/non-seizure tuples to txt file
def writeToFile(seizure_tuples, others_tuples, split=None):
    '''
    Args:
        seizure_tuples: tuples for seizure files (file_name, seizure_class)
        others_tuples: tuples for other files, e.g. other types of seizures or non-seizures (file_name, -1)
        split: "train" or "test" (before splitting train into train/dev); If None, only write into one file for each of seizure/others
    '''
    
    if split is None:
        sz_f = open(SEIZURE_FILE, 'w+')
        for name, sz_class, count in seizure_tuples:
            sz_f.write("%s,%s,%s\n" % (name, sz_class, count))
        sz_f.close()
    
        others_f = open(OTHERS_FILE, 'w+')
        for name, sz_class, count in others_tuples:
            others_f.write("%s,%s,%s\n" % (name, sz_class, count))
        others_f.close()
    else:
        if split == 'train':
            seizure_file = TRAIN_RAW_SEIZURE_FILE
            others_file = TRAIN_RAW_OTHERS_FILE
        else:
            seizure_file = TEST_SEIZURE_FILE
            others_file = TEST_OTHERS_FILE
            
        sz_f = open(seizure_file, 'w+')
        for name, sz_class, count in seizure_tuples:
            sz_f.write("%s,%s,%s\n" % (name, sz_class, count))
        sz_f.close()
    
        others_f = open(others_file, 'w+')
        for name, sz_class, count in others_tuples:
            others_f.write("%s,%s,%s\n" % (name, sz_class, count))
        others_f.close()
        

# TODO: Modify this function to split into train/dev set WITHOUT overlapping patients!!
def getSeizureTuples(filepath, verbose = True, split = None):
    """
    Args:
        filepath: path to EEG signals files
        verbos: if True, print warnings
        split: 'train' or 'test'
    Returns:
        write (file_name, seizure_class, seizure_index) to text file "seizure_files.txt" if the file contains seizure;
        write (file_name, -1, -1) to text file "others_files.txt" if the file does not contain seizure
    """
    seizure_tuples = []
    others_tuples = []
    
    patient_folders = [x[0] for x in os.walk(filepath)]
    random.shuffle(patient_folders)
    file_names = [y for x in patient_folders for y in glob(os.path.join(x, '*.edf'))]
    for i in tqdm(range(len(file_names))):
        curr_file_name = file_names[i]
        with pyedflib.EdfReader(curr_file_name) as edf:
            ordered_channels = getOrderedChannels(curr_file_name, verbose, edf.getSignalLabels())
            seizure_class = getSeizureClass(curr_file_name, "edf")
            seizure_times = getSeizureTimes(curr_file_name, "edf")
            assert len(seizure_class) == len(seizure_times)
            if len(seizure_times) > 0:
                for i in range(len(seizure_times)):
                    seizure_tuples.append((curr_file_name, seizure_class[i], i))
            else:
                others_tuples.append((curr_file_name, -1, -1))
                
    writeToFile(seizure_tuples, others_tuples, split)
    

def sliceEpoch(ordered_channels, signals, freq, slice_time, clip_len = 12):
    """
    Args:
        orderedChannels: channel index based on our order in INCLUDED_CHANNELS
        signals: 2D array of shape (num of channels, num of samples) 
        freq: sampling frequency for the current sample
        slice_time: -1 if no seizure, else start_sec of seizure in seconds
        clip_len: extracted length of EEG segments, in seconds
    Returns:
        slice_matrix: numpy array of shape (number of channels, MAX_FREQ * clip_len), e.g. (19, 3000)
    """
    signals = signals[ordered_channels, :] # extract only the channels of interest
    
    slice_matrix = []
    max_freq = MAX_FREQ
    
    if slice_time == -1: # if no seizure
        max_start = max(signals.shape[1] - freq * clip_len, 0)
        slice_time = random.randint(0, max_start)
        slice_time /= freq
    else: # if seizure
        slice_time = float(slice_time)
    
    offset = random.uniform(0, clip_len/3)

    start_time = int(freq * max(0, slice_time - offset))
    end_time = start_time + int(freq * clip_len)
    #curr_slice = signals[ch, start_time:end_time] # 1D array (start_time:end_time, )
    curr_slice = signals[:, start_time:end_time] # 2D array (number of channels, start_time:end_time)

    # pad to maximum length (max_freq * clip_len)
    diff = max_freq * clip_len - curr_slice.shape[1]
    if diff > 0:
        zeros = np.zeros((signals.shape[0], diff))
        curr_slice = np.concatenate((curr_slice, zeros), axis=1)
        
    #slice_matrix.append(curr_slice.reshape(1,-1))
             
    #slice_matrix = np.concatenate(slice_matrix, axis=0)
    slice_matrix = curr_slice
    #print('slice_matrix shape:{}'.format(slice_matrix.shape))
    
    return slice_matrix


def parseTxtFiles(seizure_file, num_folds = 5, fold_idx = 0, cross_val = True, split_train_dev = True):
    """
    Args:
        seizure_file: txt file containing list of seizure files
        num_folds: Number of folds for cross-validation
        fold_idx: Which fold?
        cross_val: whether perform cross-validation or not
        split_train_dev: For non-cv only, whether further split into train & dev sets
    Returns:
        train_tuples: tuples of (train_file_name, seizure_class, seizure_idx)
        test_tuples: tuples of (test_file_name, seizure_class, seizure_idx)
    """
    with open(seizure_file, "r") as f:
        seizure_str = f.readlines()
    
    seizure_classes = []
    seizure_patients = []
    # get seizure classes
    for i in range(len(seizure_str)):
        tup = seizure_str[i].strip('\n').split(',')
        seizure_classes.append(int(tup[1]))
        seizure_patients.append(tup[0][-22:-14]) # HARD-CODED
    seizure_patients = list(set(seizure_patients)) # get unique list of patients
    
    # stratified K-fold
    if cross_val:
        train_tuples_folds = []
        test_tuples_folds = []
        X = np.zeros((len(seizure_str))) # placeholder instead of actual X
        y = np.asarray(seizure_classes)
        skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)
        
        train_indices = []
        test_indices = []
        idx = 0
        for curr_train_idx, curr_test_idx in skf.split(X, y):
            if idx == fold_idx:
                train_idx = np.asarray(curr_train_idx)
                test_idx = np.asarray(curr_test_idx)
                idx += 1
            else:
                idx += 1
                pass
        train_str = [seizure_str[idx] for idx in train_idx]
        test_str = [seizure_str[idx] for idx in test_idx]
            
        random.shuffle(train_str)
        random.shuffle(test_str)
            
        train_tuples = []
        for i in range(len(train_str)):
            tup = train_str[i].strip("\n").split(",")
            tup[1] = int(tup[1])
            tup[2] = int(tup[2])
            train_tuples.append(tup)
        test_tuples = []
        for i in range(len(test_str)):
            tup = test_str[i].strip("\n").split(",")
            tup[1] = int(tup[1])
            tup[2] = int(tup[2])
            test_tuples.append(tup)
        
        return train_tuples, test_tuples

    else: # no cross-validation, split into train and dev with 8:2 by patients, make sure train and dev have no overlapping patients
        if split_train_dev: # if need to further split into train & dev sets
            X = seizure_patients
            print('Total number of patients with seizures: {}'.format(len(seizure_patients)))
            train_patient, test_patient = train_test_split(X, test_size=DEV_PROP, shuffle=True, random_state=1234)       
            print('Train split size:{}'.format(len(train_patient)))
            print('Dev split size:{}'.format(len(test_patient)))
            
            train_tuples = []
            test_tuples = []
            for i in range(len(seizure_str)):
                tup = seizure_str[i].strip("\n").split(",")
                tup[1] = int(tup[1])
                tup[2] = int(tup[2])
                curr_pat = tup[0][-22:-14]
                if curr_pat in train_patient:
                    train_tuples.append(tup)
                else:
                    test_tuples.append(tup)
                    
            return train_tuples, test_tuples
        else: # if no need to split into train & dev sets
            tuples = []
            for i in range(len(seizure_str)):
                tup = seizure_str[i].strip("\n").split(",")
                tup[1] = int(tup[1])
                tup[2] = int(tup[2])
                tuples.append()
            
            return tuples, []


# Function to compute time-frequency representation S
def computeTimeFreqRep(signals, freq = 12, overlap = 0.25, window = 1):
    """
    Args:
        signals: EEG segment, shape (number of channels, number of data points)
        freq: sampling frequency in Hz
        overlap: proportion of overlap with previous segment
        window: window size in seconds
    Returns:
        S: time-frequency representation, shape (number of channels, number of data points)
        where number of data points depends on freq, overlap and window.
    """
    n_ch, t_len = signals.shape

    start_time = 0
    end_time = min(t_len, start_time + int(freq * window))
      
    signal_segs = [signals[:,start_time:end_time]]

    while end_time < t_len:        
        offset = int(freq * window * overlap)
        start_time = end_time - offset
        end_time = min(t_len, start_time + int(freq * window))
        curr_seg = signals[:, start_time:end_time]
        if curr_seg.shape[1] < int(freq * window):
            diff = int(freq * window) - curr_seg.shape[1]
            curr_seg = np.concatenate((curr_seg, np.zeros((n_ch, diff))), axis=1)
        signal_segs.append(curr_seg)
    
    signal_segs = np.concatenate(signal_segs, axis=1)
    
    # FFT
    S = computeFFT(signal_segs)

    return S


def computeFFT(signals):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
    Returns:
        S: FFT of signals, (number of channels, number of data points)
    """
    N = signals.shape[1]

    # fourier transform
    fourier_signal = fft(signals, axis=1)
    
    S = 1.0/N * np.abs(fourier_signal)
    
    return S


# Function to compute divergence info of EEG data in time-freq domain (D in paper)
def computeDivergenceMap(S):
    """
    Args:
        S: time-frequency representation from computeTimeFreqRep, shape (number of data points, number of channels)
    Returns:
        D: divergence map, shape (number of data points, number of channels)
    """
    D_x = np.zeros(S.shape)
    D_y = np.zeros(S.shape)

    for i in range(S.shape[0]):
        D_x[i,:] = diff(S[i,:])

    
    for j in range(S.shape[1]):
        D_y[:,j] = diff(S[:,j])

    D = D_x + D_y
    
    return D

# Function to compute feature map I
def computeTransformI(D, smooth_window = 3):
    """
    Args:
        D: divergence map, shape (number of data points, number of channels)
        smooth_window: window size, should be an odd number
    Returns:
        I: feature map I, shape (number of data points, number of channels)
    """
    window = np.ones((smooth_window, smooth_window)) / (smooth_window**2)
    
    I = convolve2d(D, window, 'same')
            
    return I

# Compute DivSpec features
def computeDivSpec(signals, freq = 12, overlap = 0.25, window = 1):
    """
    Args:
        signals: EEG segment, shape (number of channels, number of data points)
    Returns:
        Ds: DivSpec feature maps, shape (224, 224, 3)
    """
    
    # S
    S = computeTimeFreqRep(signals, freq=freq, overlap=overlap, window=window)
    #S = computeFFT(signals)
    
    # D
    D = computeDivergenceMap(S)
    
    # I
    I = computeTransformI(D, smooth_window=SMOOTH_W)
    
    Ds = np.stack([S, D, I], axis = 2)
    Ds_res = cv2.resize(Ds, dsize=(224, 224)) # (224, 224, 3)
    Ds_res = 225 * (Ds_res - np.amin(Ds_res)) / (np.amax(Ds_res) - np.amin(Ds_res))
    
    #Ds_norm = 255 * (Ds - np.amin(Ds)) / (np.amax(Ds) - np.amin(Ds))
    
    #Ds_res = cv2.resize(Ds_norm, dsize=(224, 224)) # resize to (224, 224, 3)
    #Ds_res = Ds_norm
        
    return Ds_res
    
    
def denseSampling(ordered_channels, signals, freq, w, o):
    """
    Args:
        orderedChannels: channel index based on our order in INCLUDED_CHANNELS
        signals: 2D array of shape (num of channels, signal time length) 
        freq: sampling frequency in Hz
        w: window size in seconds
        o: overlap percentages (float)
    Returns:
        Ds_dense: all densely sampled features, numpy array of shape (number of dense samples, 224, 224, 3)
    """
    
    # re-order channels
    signals = signals[ordered_channels, :]
    
    # Get features
    Ds_dense = computeDivSpec(signals, freq=freq, overlap=o, window=w)
     
    return Ds_dense