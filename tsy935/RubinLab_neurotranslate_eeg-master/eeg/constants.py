import numpy as np

# Channels of interest
INCLUDED_CHANNELS = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG FZ', 'EEG CZ', 'EEG PZ']

# Raw data directory
DATA_DIR = "/mnt/data1/eegdbs/TUH/temple/tuh-sz-v1.2.0/v1.2.0"
TRAIN_DATA_DIR = "/mnt/data1/eegdbs/TUH/temple/tuh-sz-v1.2.0/v1.2.0/train"
TEST_DATA_DIR = "/mnt/data1/eegdbs/TUH/temple/tuh-sz-v1.2.0/v1.2.0/eval"

# Maximum sampling frequency in the dataset in Hz
MAX_FREQ = 512

# train vs dev = 9:1, if no cross validation
DEV_PROP = 0.1
loc = "/mnt/home2/dlongo/eegML/tsy935/RubinLab_neurotranslate_eeg-master/eeg/"
# File names
# For seizure type classification
SEIZURE_FILE = loc + "data/seizure_files.txt"
OTHERS_FILE = loc + "data/others_files.txt"


TRAIN_RAW_SEIZURE_FILE = loc + "data/train_seizure_files.txt"
TRAIN_RAW_OTHERS_FILE = loc + "data/train_others_files.txt"

TRAIN_SEIZURE_FILE = loc + "data/trainSet_seizure_files.txt"
DEV_SEIZURE_FILE = loc + "data/devSet_seizure_files.txt"
TEST_SEIZURE_FILE = loc + "data/testSet_seizure_files.txt"
TRAIN_OTHERS_FILE = loc + "data/trainSet_others_files.txt"
DEV_OTHERS_FILE = loc + "data/devSet_others_files.txt"
TEST_OTHERS_FILE = loc + "data/testSet_others_files.txt"

# Preprocessed files
H5_TRAIN_DIR = "/mnt/home2/dlongo/input/nonOverlap/train_features.h5"
H5_DEV_DIR = "/mnt/home2/dlongo/input/nonOverlap/dev_features.h5"
H5_TEST_DIR = "/mnt/home2/dlongo/input/nonOverlap/test_features.h5"

### H5_CV_DIR = "/share/pi/rubin/siyitang/eeg/input/cv" ### Don't have cross validaton data

# For seizure detection
ALL_SEIZURE_FILE = loc + "data/all_seizures.txt"
NONSEIZURE_FILE = loc + "data/all_nonseizures.txt"

# args file name
ARGS_FILE_NAME = 'args.json'

# clip length for each EEG segment, in seconds
#CLIP_LEN = 12

# dense feature sampling parameters
#Fs = [12, 24, 48, 64, 96]
#Ws = [1, 2, 4, 8, 16]
#Os = [0.25, 0.50, 0.75]
Fs = [24, 48, 96]
Ws = [2, 4, 16]
Os = [0.25]

DENSE_PARAMS = []
for f in Fs:
    for w in Ws:
        for o in Os:
            DENSE_PARAMS.append([f, w, o])

# window length and overlap at test time
# TODO: Update these, see author's reply!!
test_w = 2
test_o = 0.25

# smoothing window size for computing feature map I
SMOOTH_W = 8

# number of seizure classes
NUM_CLASSES = 7
TARGET_LABS = ['fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz']
LABEL_DICT = {'fnsz': 0, 'gnsz': 1, 'spsz': 2, 'cpsz': 3, 'absz': 4, 'tnsz': 5, 'tcsz': 6}

# set random seed
SEED = 123

# Configuration of dense net:
BLOCK_CONFIG = (6, 12, 32, 32)
NUM_INIT_FEATURES = 64 # ???
BN_SIZE = 4

# weights for different classes used in loss function, because of imbalanced classes
CLASS_W = np.asarray([0.9, 4.5, 10.0, 4.1, 10.0, 10.0, 10.0])
#CLASS_W = np.asarray([1., 1., 1., 1., 1., 1., 1.])