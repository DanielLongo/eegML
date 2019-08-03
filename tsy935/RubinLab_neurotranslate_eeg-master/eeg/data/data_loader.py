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
# from data.data_utils import *

# Add parent dir to the sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from constants import *


class SeizureDataset(Dataset):  
    def __init__(self, file_dir=TRAIN_SEIZURE_FILE, num_folds=5, fold_idx=None, cross_val=False, split='train', test_freq=96, transform=None): #, reduce_2nd_dim=False):
        """
        Args:
            file_dir: (string) directory containing the list of file names to pull in
            num_folds: Number of folds in cross-validation
            fold_idx: Index of test fold in cross-validation
            cross_val: Whether to perform cross-validation
            split: 'train', 'dev' or 'test' if no cross-validation; else 'train' or 'test'
            test_freq: Sampling freqency at test time
        """
        if cross_val and (split == 'dev'):
            raise ValueError('For cross-validation, no dev set required.')
                
        self.split = split
        self.file_dir = file_dir
        self.test_freq = test_freq
        self.transform = transform
        print('file dir:{}'.format(file_dir))
        
        if cross_val:
            if split == 'train':
                self.h5_dir = os.path.join(H5_CV_DIR, 'fold' + str(fold_idx) + '_train_features.h5')
            else:
                self.h5_dir = os.path.join(H5_CV_DIR, 'fold' + str(fold_idx) + '_test_features.h5')
        else:
            if split == 'train':
                self.h5_dir = H5_TRAIN_DIR
            elif split == 'dev':
                self.h5_dir = H5_DEV_DIR
            else:
                self.h5_dir = H5_TEST_DIR
        
        # Read file tuples, (file_name, seizure_class, seizure_idx)
        with open(file_dir, 'r') as f:
            seizure_str = f.readlines()
        
        file_tuples = []
        h5_file_names = []
        for i in range(len(seizure_str)):
            tup = seizure_str[i].strip("\n").split(",")
            tup[1] = int(tup[1])
            tup[2] = int(tup[2])
            file_tuples.append(tup)
            h5_file_names.append(tup[0] + '_' + str(tup[2])) # original_file_name + seizure_idx
        self.file_tuples = file_tuples
      
        # Load preprocessed data
        self.features = []
        print(self.h5_dir)
        with h5py.File(self.h5_dir, 'r') as hf:
            for f_name in h5_file_names:
                # print("f_name", f_name)
                self.features.append(hf[f_name][()])

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (feature, seizure_class, curr_file_name)
        """
        curr_file_name, seizure_class, seizure_idx = self.file_tuples[idx]
        
        feature = torch.FloatTensor(self.features[idx]) # (number of dense samples, 224, 224, 3) or (224, 224, 3)
        
        if self.split == 'train':
            n_dense = feature.shape[0]
            feature = feature.permute(0, 3, 1, 2) # (number of dense samples, 3, 224, 224)
        else:
            n_dense = 1
            feature = feature.permute(2, 0, 1)
            
        #print('Feature shape: {}'.format(feature.size()))       
        
        label = torch.LongTensor([seizure_class] * n_dense)
        #print('Label shape: {}'.format(label.size()))
        
        writeout_file_name = curr_file_name + '_' + str(seizure_idx)
        if self.transform is None:
            return (feature, label, writeout_file_name)
        else:
            return (self.transform(feature), label, writeout_file_name)
# import pyedflib
# import os
# import numpy as np
# import random
# import torch
# import sys
# import inspect
# import time
# import h5py
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# # from data.data_utils import *
# from data_utils import *

# # Add parent dir to the sys path
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

# from constants import *


# class SeizureDataset(Dataset):  
#     def __init__(self, file_dir, num_folds=5, fold_idx=None, cross_val=False, split='train', test_freq=96):
#         """
#         Args:
#             file_dir: (string) directory containing the list of file names to pull in
#             num_folds: Number of folds in cross-validation
#             fold_idx: Index of test fold in cross-validation
#             cross_val: Whether to perform cross-validation
#             split: 'train', 'dev' or 'test' if no cross-validation; else 'train' or 'test'
#             test_freq: Sampling freqency at test time
#         """
#         if cross_val and (split == 'dev'):
#             raise ValueError('For cross-validation, no dev set required.')
                
#         self.split = split
#         self.file_dir = file_dir
#         self.test_freq = test_freq
#         print('file dir:{}'.format(file_dir))
        
#         if cross_val:
#             if split == 'train':
#                 self.h5_dir = os.path.join(H5_CV_DIR, 'fold' + str(fold_idx) + '_train_features.h5')
#             else:
#                 self.h5_dir = os.path.join(H5_CV_DIR, 'fold' + str(fold_idx) + '_test_features.h5')
#         else:
#             if split == 'train':
#                 self.h5_dir = H5_TRAIN_DIR
#             elif split == 'dev':
#                 self.h5_dir = H5_DEV_DIR
#             else:
#                 self.h5_dir = H5_TEST_DIR
        
#         # Read file tuples, (file_name, seizure_class, seizure_idx)
#         with open(file_dir, 'r') as f:
#             seizure_str = f.readlines()
        
#         file_tuples = []
#         h5_file_names = []
#         for i in range(len(seizure_str)):
#             tup = seizure_str[i].strip("\n").split(",")
#             tup[1] = int(tup[1])
#             tup[2] = int(tup[2])
#             file_tuples.append(tup)
#             h5_file_names.append(tup[0] + '_' + str(tup[2])) # original_file_name + seizure_idx
#             # print(h5_file_names)
#         self.file_tuples = file_tuples
      
#         # Load preprocessed data
#         self.features = []
#         print(self.h5_dir)
#         with h5py.File(self.h5_dir, 'r') as hf:
#             for f_name in h5_file_names:
#                 print(f_name)
#                 self.features.append(hf[f_name][()])

#     def __len__(self):
#         return len(self.file_tuples)

#     def __getitem__(self, idx):
#         """
#         Args:
#             idx: (int) index in [0, 1, ..., size_of_dataset-1]
#         Returns:
#             a tuple of (feature, seizure_class, curr_file_name)
#         """
#         curr_file_name, seizure_class, seizure_idx = self.file_tuples[idx]
        
#         feature = torch.FloatTensor(self.features[idx]) # (number of dense samples, 224, 224, 3) or (224, 224, 3)
        
#         if self.split == 'train':
#             n_dense = feature.shape[0]
#             feature = feature.permute(0, 3, 1, 2) # (number of dense samples, 3, 224, 224)
#         else:
#             n_dense = 1
#             feature = feature.permute(2, 0, 1)
            
#         # print('Feature shape: {}'.format(feature.size()))       
        
#         label = torch.LongTensor([seizure_class] * n_dense)
#         # print('Label shape: {}'.format(label.size()))
        
#         writeout_file_name = curr_file_name + '_' + str(seizure_idx)
        
#         return (feature, label, writeout_file_name)

if __name__ == "__main__":
    seizure_file = TRAIN_SEIZURE_FILE
    train_dataset = SeizureDataset(seizure_file) #, num_folds=args.num_folds, fold_idx=fold_idx, cross_val=cross_val, split='train')
    train_loader = DataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=64)
    print("length of train_loader", len(train_loader))
    for features, y, _, in train_loader:
        print("feature shape", features.shape)
        print("max", np.max(features.view(-1).numpy())) #, dim=0))
        print("min", np.min(features.view(-1).numpy())) #, dim=0))
        print("shape", np.shape(features.view(-1).numpy()))
        print("y", y.shape)
