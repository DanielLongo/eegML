import numpy as np 
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