import mne
import numpy as np
from mne.datasets import sample
import matplotlib.pyplot as plt
import h5py

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_fname) 
fwd = mne.read_forward_solution("sample_forward_model")

fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                         use_cps=True)
def save_data(data, filename, channel_names, frequency):
	info = mne.create_info(channel_names, frequency, 
		"eeg")
	customraw = mne.io.RawArray(data, info)
	customraw.save(filename, overwrite=True)

def generate_data(num_examples, data_dir, fwd_fixed, n_sensors=60, fs_gen=200, n_times=10, frequency=200, prefix="generated-"):
	leadfield = fwd_fixed['sol']['data']

	n_dipoles = leadfield.shape[1]
	vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]

	time_step = 1.0/fs_gen # sample freq = 200 was 0.5
	n_times = 10 * fs_gen  # try 10 s of generation
	channel_names = [str(i) for i in range(60)]

	for i in range(num_examples):
		z = np.dot(np.random.randn(n_dipoles, n_sensors), np.random.randn(n_sensors, n_times)) * 1e-9
		filename = data_dir + "/"  + prefix + str(i) + "raw.fif"
		stc = mne.SourceEstimate(z, vertices, tmin=0., tstep=time_step)
		leadfield = mne.apply_forward(fwd_fixed, stc, info).data
		save_data(leadfield, filename, channel_names, frequency)
		print("i: " + str(i) + " - " + str(int((i+1) * 100/num_examples)) + "%")
	print("finished")
	
data_dir = "../estimated_clean_eegs"
generate_data(10, data_dir , fwd_fixed)


