import mne
import time
import datetime
import numpy as np
import os
import torch
from torch.utils import data
import h5py
import random
import sklearn
from sklearn import preprocessing
from mne.datasets import sample
import sys, os

class EstimatedEEGs(data.Dataset):
	def __init__(self, num_channels=19, length=1000, batch_size=64):
		self.batch_size = batch_size
		self.length = length
		self.num_nodes = num_channels
		self.fs_gen = 250
		self.time_step = 1.0/self.fs_gen

		data_path = sample.data_path()
		self.raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
		fwd = mne.read_forward_solution("./forward_model/sample_forward_model")
		self.fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
		leadfield = self.fwd_fixed['sol']['data']
		self.n_dipoles = leadfield.shape[1]
		self.vertices = [src_hemi['vertno'] for src_hemi in self.fwd_fixed['src']]
		self.channel_names = [str(x) for x in range(self.num_nodes)]

	def __len__(self):
		return -1 #length unlimited 

	def __getitem__(self, index):
		noise = torch.randn(self.batch_size, self.n_dipoles, self.length) * 1e-9
		sys.stdout = open(os.devnull, 'w') #disables console logs
		estimated = self.apply_forward_model(noise)
		sys.stdout = sys.__stdout__ #enables console logs
		return estimated

	def apply_forward_model(self, x):
		out = []
		for z in x:
			info = mne.io.read_info(self.raw_fname)
			stc = mne.SourceEstimate(z.cpu().detach().numpy(), self.vertices, tmin=0., tstep=self.time_step)
			leadfield = mne.apply_forward(self.fwd_fixed, stc, info).data[0:self.num_nodes]
			out += [leadfield]
		out = np.asarray(out)
		out = torch.from_numpy(out).transpose(1,2)
		if x.is_cuda:
			out = out.type(torch.cuda.FloatTensor)
		else:
			out = out.type(torch.FloatTensor)
		# print("OUTOUTOUT", out.shape)
		# (n sensors, n samples)
		return out

if __name__ == "__main__":
	dataset = EstimatedEEGs(num_channels=44, length=1004)
	# dataset.shuffle()
	print(dataset[1000].shape)