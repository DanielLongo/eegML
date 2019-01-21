import mne
import numpy as np
from mne.datasets import sample
import matplotlib.pyplot as plt
import h5py

filename = "./generated_eegs/from_froward_model/test"
n_sensors = 60
fs_gen = 200
time_step = 1.0/fs_gen # sample freq = 200 was 0.5
n_times = 10 * fs_gen  # try 10 s of generation
frequency = 200
n = 10

channel_names = [str(i) for i in range(60)]

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_fname) 
fwd = mne.read_forward_solution("sample_forward_model")
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
leadfield = fwd_fixed['sol']['data']
n_dipoles = leadfield.shape[1]
vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]
info = mne.create_info(channel_names, frequency, "eeg")

for i in range(n):
	z = np.dot(np.random.randn(n_dipoles, n_sensors), np.random.randn(n_sensors, n_times)) * 1e-9
	stc = mne.SourceEstimate(z, vertices, tmin=0., tstep=time_step)
	leadfield = mne.apply_forward(fwd_fixed, stc, info).data
	customraw = mne.io.RawArray(leadfield.data, info)
	customraw.save(filename + "-" + str(i) + ".fif", overwrite=True)
