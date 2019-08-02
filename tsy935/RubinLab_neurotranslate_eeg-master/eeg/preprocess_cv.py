import numpy as np
import h5py
from data.data_utils import *
import os

# set seed
SEED = 123
np.random.seed(SEED)

# Define some constants here
output_dir = '/share/pi/rubin/siyitang/eeg/input/cv_full'
txt_dir = 'data'
Fs = [12, 24, 48, 64, 96]
Ws = [1, 2, 4, 8, 16]
Os = [0.25, 0.50, 0.75]
DENSE_PARAMS = []
for f in Fs:
    for w in Ws:
        for o in Os:
            DENSE_PARAMS.append([f, w, o])
TRAIN_INCREASES = len(Fs) * len(Ws) * len(Os)
print('Train set increases by {}'.format(TRAIN_INCREASES))
test_w = 2
test_o = 0.25
NUM_FOLDS = 5
SEIZURE_FILE = "data/seizure_files.txt"

def main(num_folds):
    """
    Perform feature extraction before-hand.
    Increase the sample size by different combinations of sampling freq, window size & overlap.
    """
    for fold_idx in range(num_folds):
        print('Preprocessing for fold{}...'.format(fold_idx))
        train_write_txt = os.path.join(txt_dir,'fold' + str(fold_idx) + '_trainSet_seizure_files.txt')
        test_write_txt = os.path.join(txt_dir, 'fold' + str(fold_idx) + '_testSet_seizure_files.txt')
        
        # Split into train/test, stratified K fold
        file_tuples = parseTxtFiles(SEIZURE_FILE,  num_folds=num_folds, fold_idx=fold_idx, cross_val=True)
        train_file_tuples = file_tuples[0]
        test_file_tuples = file_tuples[1]
        
        # Write into a new text file for train set only
        f_train = open(train_write_txt, 'w+')
        for name, sz_class, count in train_file_tuples:
            f_train.write("%s,%s,%s\n" % (name, sz_class, count))
        f_train.close()
        
        f_test = open(test_write_txt, 'w+')
        for name, sz_class, count in test_file_tuples:
            f_test.write("%s,%s,%s\n" % (name, sz_class, count))
        f_test.close()

        ##### TRAIN SET #####
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
        train_h5_file = os.path.join(output_dir, 'fold' + str(fold_idx) + '_train_features.h5')
        with h5py.File(train_h5_file, 'w') as hf:
            for key, val in features.items():
                hf.create_dataset(key,  data = val)
                
        ##### TEST SET #####
        for idx in range(len(test_file_tuples)):
            curr_file_name, seizure_class, seizure_idx = test_file_tuples[idx]
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
        test_h5_file = os.path.join(output_dir, 'fold' + str(fold_idx) + '_test_features.h5')
        with h5py.File(test_h5_file, 'w') as hf:
            for key, val in features.items():
                hf.create_dataset(key,  data = val)

if __name__ == '__main__':
    main(NUM_FOLDS)