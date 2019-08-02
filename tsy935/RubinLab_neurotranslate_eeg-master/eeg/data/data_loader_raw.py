import pyedflib
import os
import numpy as np
import random
import torch
import sys
import inspect
import time
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
# from data.data_utils import *
from data_utils import *

# Add parent dir to the sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from constants import *


class SeizureDataset(Dataset):  
    def __init__(self, file_dir, num_folds=5, fold_idx=None, cross_val=False, split='train', test_freq=96, num_channels=1, length=768, num_examples=-1, onehot=True):
        """
        Args:
            file_dir: (string) directory containing the list of file names to pull in
            num_folds: Number of folds in cross-validation
            fold_idx: Index of test fold in cross-validation
            cross_val: Whether to perform cross-validation
            split: 'train', 'dev' or 'test' if no cross-validation; else 'train' or 'test'
            test_freq: Sampling freqency at test time
        """
        self.num_channels = num_channels
        self.length = length
        self.raw_signals = []
        self.seizure_classes = []
        self.num_examples = num_examples

        if split == 'dev':
            raise ValueError("Invalid split option. Choose from 'train' or 'test'.")
        
        if split == 'train':
            file_dir = TRAIN_RAW_SEIZURE_FILE
            file_tuples = parseTxtFiles(file_dir,  num_folds=None, fold_idx=None, cross_val=False, split_train_dev=True)
            self.train_file_tuples = file_tuples[0]
            self.dev_file_tuples = file_tuples[1]
            self.load_train()
            
                  
        if split == "test":
            # Since test set does not change, no need to write into a new file anymore
            file_dir = TEST_SEIZURE_FILE
            file_tuples = parseTxtFiles(file_dir,  num_folds=None, fold_idx=None, cross_val=False, split_train_dev=False)
            self.test_file_tuples = file_tuples[0]    
            self.load_test()

        assert(len(self.raw_signals) == len(self.seizure_classes))

        self.raw_signals = torch.LongTensor(self.raw_signals)
        self.seizure_classes = torch.LongTensor(self.seizure_classes)
        if onehot:
            self.convert_seizure_classes_to_onehot()

    def convert_seizure_classes_to_onehot(self, num_classes=8):
        # self.raw_signals = F.one_hot(self.raw_signals, num_classes=num_classes)
        pass
        
    def load_train(self):            
        ##### TRAIN SET #####
        print('Loading train set...')
        features = {}        
        for idx in range(len(self.train_file_tuples)):
            if len(self.raw_signals) == self.num_examples:
                break
            curr_file_name, seizure_class, seizure_idx = self.train_file_tuples[idx]
            # read file
            # print("curr_file_name", curr_file_name)
            # currr_file = curr_file_name.split(".")[0] + ".eeg.h5"
            f = pyedflib.EdfReader(curr_file_name)
        
            ordered_channels = getOrderedChannels(curr_file_name, False, f.getSignalLabels())
       
            signals = getEDFsignals(f)
                
            frequencies = getSamplingFreq(f, ordered_channels)
            f._close()

            freq = frequencies[0]        
            
            seizure_times = getSeizureTimes(curr_file_name, file_type="edf")
            seizure_times = seizure_times[seizure_idx]
            start_t = int(freq * seizure_times[0])
            end_t = int(freq * seizure_times[1])
            curr_signals = signals[:, start_t:end_t]
            curr_length = curr_signals.shape[1]
            if curr_signals.shape[0] < self.num_channels or curr_length < self.length * 3:
                continue
            # curr_mid = math.floor(curr_length / 2)
            # curr_signals = curr_signals[:self.num_channels, curr_mid - self.length: curr_mid]
            curr_signals = curr_signals[:self.num_channels, :self.length]
                                        
            self.raw_signals.append(curr_signals)
            self.seizure_classes.append(seizure_class)

    def load_dev(self):
        ##### DEV SET #####
        print('Loading dev set...')
        features = {}
        for idx in range(len(self.dev_file_tuples)):
            if len(self.raw_signals) == self.num_examples:
                break
            curr_file_name, seizure_class, seizure_idx = self.dev_file_tuples[idx]
            print(curr_file_name)
            # read file
            f = pyedflib.EdfReader(curr_file_name)
        
            ordered_channels = getOrderedChannels(curr_file_name, False, f.getSignalLabels())
       
            signals = getEDFsignals(f)
                    
            frequencies = getSamplingFreq(f, ordered_channels)
            freq = frequencies[0]        
        
            seizure_times = getSeizureTimes(curr_file_name, file_type="edf")
            seizure_times = seizure_times[seizure_idx]
            start_t = int(freq * seizure_times[0])
            end_t = int(freq * seizure_times[1])
            curr_signals = signals[:, start_t:end_t]
            curr_length = curr_signals.shape[1]
            if curr_signals.shape[0] < self.num_channels or curr_length < self.length * 3:
                continue
            # curr_mid = math.floor(curr_length / 2)
            # curr_signals = curr_signals[:self.num_channels, curr_mid - self.length: curr_mid]
            curr_signals = curr_signals[:self.num_channels, :self.length]
                                        
            f._close()
            self.raw_signals.append(curr_signals)
            self.seizure_classes.append(seizure_class)

    def load_test(self):
        ##### TEST SET #####
        print('Loading test set...')        
        features = {}
        for idx in range(3):
            #for idx in range(len(test_file_tuples)):
            if len(self.raw_signals) == self.num_examples:
                break
            curr_file_name, seizure_class, seizure_idx = self.test_file_tuples[idx]
            print(curr_file_name)
            # read file
            f = pyedflib.EdfReader(curr_file_name)
        
            ordered_channels = getOrderedChannels(curr_file_name, False, f.getSignalLabels())
       
            signals = getEDFsignals(f)
                    
            frequencies = getSamplingFreq(f, ordered_channels)
            freq = frequencies[0]        
        
            seizure_times = getSeizureTimes(curr_file_name, file_type="edf")
            seizure_times = seizure_times[seizure_idx]
            start_t = int(freq * seizure_times[0])
            end_t = int(freq * seizure_times[1])
            curr_signals = signals[:, start_t:end_t]
            curr_length = curr_signals.shape[1]
            if curr_signals.shape[0] < self.num_channels or curr_length < self.length * 3:
                continue
            # curr_mid = math.floor(curr_length / 2)
            # curr_signals = curr_signals[:self.num_channels, curr_mid - self.length: curr_mid]
            curr_signals = curr_signals[:self.num_channels, :self.length]
                                        
            f._close()
            self.raw_signals.append(curr_signals)
            self.seizure_classes.append(seizure_class)

    def __len__(self):
        assert(len(self.raw_signals) == len(self.seizure_classes))
        return len(self.raw_signals)

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (feature, seizure_class, str:None)
        """
        return (self.raw_signals[idx], self.seizure_classes[idx], "NONE")

if __name__ == "__main__":
    seizure_file = TRAIN_SEIZURE_FILE
    train_dataset = SeizureDataset(seizure_file, num_channels=33, length=768, num_examples=32*2) #, num_folds=args.num_folds, fold_idx=fold_idx, cross_val=cross_val, split='train')
    # x = train_dataset[0]
    # print(x[0].size())
    train_loader = DataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=32,)
    print("length of train_loader", len(train_loader))
    print("len of train_loader", len(train_loader))
    for features, y, _, in train_loader:
        print("feature shape", features.shape)
        np.save("tesing_raw_dataloader", features)
        print("y", y.shape)
        print(y)

