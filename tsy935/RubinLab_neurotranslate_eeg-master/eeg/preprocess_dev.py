import numpy as np
import h5py
from data.data_utils import *
from constants import *
import os

# set seed
SEED = 123
np.random.seed(SEED)

output_dir = '/mnt/home2/dlongo/input/nonOverlap/'


TRAIN_INCREASES = len(Fs) * len(Ws) * len(Os)

def main(split):
    """
    Perform feature extraction before-hand.
    Increase the sample size by different combinations of sampling freq, window size & overlap.
    Args:
        split: "train" or "test", if "train", will be further split into train and dev splits
    """
    if split == 'dev':
        raise ValueError("Invalid split option. Choose from 'train' or 'test'.")
    
    if split == 'train':
        file_dir = TRAIN_RAW_SEIZURE_FILE
        file_tuples = parseTxtFiles(file_dir,  num_folds=None, fold_idx=None, cross_val=False, split_train_dev=True)
        train_file_tuples = file_tuples[0]
        dev_file_tuples = file_tuples[1]
        
        # Write into a new text file for train set
        new_f = open(TRAIN_SEIZURE_FILE, 'w+')
        for name, sz_class, count in train_file_tuples:
            new_f.write("%s,%s,%s\n" % (name, sz_class, count))
        new_f.close()
                        
        # Write into a new text file for dev set
        new_f = open(DEV_SEIZURE_FILE, 'w+')
        for name, sz_class, count in dev_file_tuples:
            new_f.write("%s,%s,%s\n" % (name, sz_class, count))
        new_f.close()
              
    else:
        # Since test set does not change, no need to write into a new file anymore
        file_dir = TEST_SEIZURE_FILE
        file_tuples = parseTxtFiles(file_dir,  num_folds=None, fold_idx=None, cross_val=False, split_train_dev=False)
        test_file_tuples = file_tuples[0]    
    
    if split == 'train':
                
        ##### TRAIN SET #####
        print('Preprocessing train set...')

        # Compute features for train set
        features = {}        
        for idx in range(len(train_file_tuples)):
            curr_file_name, seizure_class, seizure_idx = train_file_tuples[idx]
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
                                        
            f._close()        
                
            # dense features, only for training split
            dense_feats = []
            for sampling_idx in range(TRAIN_INCREASES):
                dense_param = DENSE_PARAMS[sampling_idx]
                dense_feats.append(denseSampling(ordered_channels, curr_signals, dense_param[0], dense_param[1], dense_param[2]))
            
            write_file_name = curr_file_name + '_' + str(seizure_idx)
            print(write_file_name)
            features[write_file_name] = dense_feats        

        # Write into h5py file
        train_h5_file = os.path.join(output_dir, 'train_features.h5')
        with h5py.File(train_h5_file, 'w') as hf:
            for key, val in features.items():
                hf.create_dataset(key,  data = val)
        
        
        ##### DEV SET #####
        print('Preprocessing dev set...')
        
        # Compute features for dev set        
        features = {}
        for idx in range(len(dev_file_tuples)):
            curr_file_name, seizure_class, seizure_idx = dev_file_tuples[idx]
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
                
            f._close()
                
            # dense features, only for training split
            write_file_name = curr_file_name + '_' + str(seizure_idx)
            print(write_file_name)
            features[write_file_name] = denseSampling(ordered_channels, curr_signals, 96, test_w, test_o) 

        # Write into h5py file
        dev_h5_file = os.path.join(output_dir, 'dev_features.h5')
        with h5py.File(dev_h5_file, 'w') as hf:
            for key, val in features.items():
                hf.create_dataset(key,  data = val)

    else:
        
        ##### TEST SET #####
        print('Preprocessing test set...')        

        # Compute features for test set
        features = {}
        for idx in range(3):
        #for idx in range(len(test_file_tuples)):
            curr_file_name, seizure_class, seizure_idx = test_file_tuples[idx]
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
                
            f._close()
        
            write_file_name = curr_file_name + '_' + str(seizure_idx)
            print(write_file_name)
            features[write_file_name] = denseSampling(ordered_channels, curr_signals, 96, test_w, test_o)

        # Write into h5py file
        test_h5_file = os.path.join(output_dir, 'test_features.h5')
        with h5py.File(test_h5_file, 'w') as hf:
            for key, val in features.items():
                hf.create_dataset(key,  data = val)
    

if __name__ == '__main__':
    main('train')