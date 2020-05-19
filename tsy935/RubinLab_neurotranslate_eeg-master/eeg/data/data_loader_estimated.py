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
from data_utils import *
sys.path.append("/mnt/home2/dlongo/eegML/data_loaders")
from forward_model_dataloader import ForwardModelDataset
# from data.data_utils import *

# Add parent dir to the sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from constants import *


def normalize(batch):
    batch = batch - batch.mean()
    batch = batch / batch.std()
    batch = batch / np.abs(batch).max()
    return batch

class EstimatedDataset(Dataset):  
    def __init__(self, num_examples, transform=None):
        """
        """
        self.transform = transform
        
        data = ForwardModelDataset(num_examples=num_examples, num_channels=20, length=1000, batch_size=1, save_source=False)
        self.estimated_eegs = data.getAllEEGs()
        # self.estimated_eegs = [normalize(x)  for x in self.estimated_eegs]
        self.processed_eegs = []
        # Div Spec params
        self.freq, self.overlap, self.window = 12, 0.25, 1
        for raw_eeg in self.estimated_eegs:
            self.processed_eegs += [self.compute_div_spec(raw_eeg)]
        
        
        assert(len(self.estimated_eegs) == len(self.processed_eegs))

    def __len__(self):
        assert(len(self.estimated_eegs) == len(self.processed_eegs))
        return len(self.estimated_eegs)

    def __getitem__(self, idx):
        return self.get_raw(idx), self.get_div_spec(idx)

    def get_div_spec(self, idx):
        out = torch.from_numpy(self.processed_eegs[idx]).type('torch.FloatTensor')
        if self.transform != None:
            out = self.transform(out)
        return out

    def get_raw(self, idx, transform=False):
        out = torch.from_numpy(self.estimated_eegs[idx]).type('torch.FloatTensor')
        if self.transform != None and transform:
            out = self.transform(out)
        return out

    def compute_div_spec(self, raw):
       # print("raw", raw)
        return computeDivSpec(raw, self.freq, self.overlap, self.window)

    @staticmethod
    def compute_div_spec(raw, freq=12, overlap=.25, window=1):
      #  print("raw", raw)
        return computeDivSpec(raw, freq, overlap, window)



if __name__ == "__main__":
    dataset = EstimatedDataset(10)
    print((dataset[0]).shape)
