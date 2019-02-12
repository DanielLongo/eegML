import numpy as np 
import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

# import eeghdf
import mne
import mne.io
import h5py
def save_EEG(data,num_channels, frequency, filename, channel_names=None):
	if data.shape[1] < 10000:
		np.save(filename, data[0, :, :])	
	else:
		np.save(filename, data[0, :10000, :])
	# channel_names = list(range(num_channels)) if (channel_names==None) else channel_names
	# info = mne.create_info(channel_names, frequency, 
                       # "eeg")
	# customraw = mne.io.RawArray(data, info)
	# customraw.save(filename, overwrite=True)

import numpy as np 
def save_EEG_tfr(ch5A, ch5D, num_channels, frequency, filename, channel_names=None, wtype='sym3'):
	recon_arr = pywt.idwt(ch5A, ch5D, wtype)
	if recon_arr.shape[1] < 10000:
		np.save(filename, recon_arr[0, :, :].reshape(1000, 44))	
	else:
		np.save(filename, recon_arr[0, :10000, :].reshape(1000, 44))