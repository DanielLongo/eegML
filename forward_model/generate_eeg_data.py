import mne
import numpy as np
from mne.datasets import sample
import matplotlib.pyplot as plt
import h5py
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_fname) 
fwd = mne.read_forward_solution("sample_forward_model")
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
leadfield = fwd_fixed['sol']['data']
n_dipoles = leadfield.shape[1]
vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]

filename = "/Users/DanielLongo/server/mnt/home2/dlongo/eegML/generated_eegs/from_forward_model/test"
n_sensors = 60
n_sensors_out = 44
fs_gen = 200
time_step = 1.0/fs_gen # sample freq = 200 was 0.5
n_times = 50 * fs_gen  # try 10 s of generation
frequency = 200
n = 10
channel_names_out = [str(i) for i in range(n_sensors_out)]
channel_names_in = [str(i) for i in range(n_sensors)]

for i in range(n):
	info = mne.io.read_info(raw_fname) 
	z = np.dot(np.random.randn(n_dipoles, n_sensors), np.random.randn(n_sensors, n_times)) * 1e-9
	stc = mne.SourceEstimate(z, vertices, tmin=0., tstep=time_step)
	leadfield = mne.apply_forward(fwd_fixed, stc, info).data
	data = leadfield.data#[:44,:]
	print(type(data))
	print("data", data.shape)
	info_in = mne.create_info(channel_names_in, frequency, 
	                       "eeg")#, montage='standard_1020')
	info_out = mne.create_info(channel_names_out, frequency, 
	                       "eeg")#, montage='standard_1020')
	customraw = mne.io.RawArray(mne.io.RawArray(data, info_in)[:44,:][0], info_out)
	# print(np.shape(customraw))
	# print(np.shape(customraw[:44,:][1]))
	customraw.save(filename + "-" + str(i) + ".fif", overwrite=True)