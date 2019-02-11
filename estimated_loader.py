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

class EstimatedEEGs(data.Dataset):
	def __init__(self, num_channels=19,length=1000):
		self.length = length
		self.num_channels = num_channels

	def __len__(self):
		return -1 #length unlimited 

	def __getitem__(self, index, batch_size=64):
		#old bad method
		noise = 
		batch_filenames = self.batched_filenames[index]
		signals, attributes = read_filenames(batch_filenames, self.length, delay=self.delay)
		sample = torch.from_numpy(np.asarray(signals))
		sample = sample.view(-1, sample.shape[2], sample.shape[1]).type('torch.FloatTensor')

		return sample

			self.num_nodes = num_nodes
		self.fs_gen = fs_gen
		self.time_step = 1.0/fs_gen

		data_path = sample.data_path()
		self.raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
		fwd = mne.read_forward_solution("./forward_model/sample_forward_model")
		self.fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
		leadfield = self.fwd_fixed['sol']['data']
		n_dipoles = leadfield.shape[1]
		self.vertices = [src_hemi['vertno'] for src_hemi in self.fwd_fixed['src']]
		self.channel_names = [str(x) for x in range(num_nodes)]


	def apply_foward_model(self, x):
		out = []
		for z in x:
			info = mne.io.read_info(self.raw_fname)
			stc = mne.SourceEstimate(z.cpu().detach().numpy(), self.vertices, tmin=0., tstep=self.time_step)
			# print("STC", stc)
			# print("Info", info)
			leadfield = mne.apply_forward(self.fwd_fixed, stc, info).data[0:self.num_nodes]
			# leadfield = torch.from_numpy(leadfield)
			# leadfield = leadfield.transpose(0,1)
			# print("YOUYOUYOUYOUY")
			# array = leadfield.to_data_frame().values
			# print("array", leadfield.shape)
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
	dataset = EstimatedEEGs(num_channels=44, length=100000)
	dataset.shuffle()
	dataset[0]