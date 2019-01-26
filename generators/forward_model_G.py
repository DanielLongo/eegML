import torch
from torch import nn
import mne
import numpy as np
from mne.datasets import sample

class ForwardModelEnabledG(nn.Module):
	#input shape (n_sensors, n_times)
	#output shape (seq_len, batch, input_size)
	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True, fs_gen=200):
		super(ForwardModelEnabledG, self).__init__()

		## For the Forward Model ##
		# (n_dipoles, n_times) -> (seq_len, input_size)
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

		## Brain Space Modifications ##
		#(n_times, n_dipoles) -> (n_times, n_dipoles)
		self.rnnBS1 = nn.LSTM(input_size=n_dipoles, hidden_size=d, num_layers=num_layers, bidirectional=bidirectional, batch_first=True) 
		self.rnnBS2 = nn.LSTM(input_size=d * (1 + (bidirectional*1)), hidden_size=n_dipoles, num_layers=1, bidirectional=False, batch_first=True)
		self.ReLU = nn.ReLU()


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

	def apply_brain_space_mods(self, x):
		# print("brain space mods", x.shape)
		out, _ = self.rnnBS1(x)
		# out = ReLU(out)
		# print("rrnBS1", out.shape)
		out, _ = self.rnnBS2(out)
		# print("rrnBS2", out.shape)
		# out = out.transpose(1,2)
		return out


	def forward(self, z):
		z *= 1e-9
		out = self.apply_brain_space_mods(z)
		out = z.transpose(1,2)
		out = self.apply_foward_model(out)
		return out

	# def generate_noise(self, batch_size, num_signals, num_dipoles=7498):
	# 	return [torch.randn(batch_size, num_signals, num_dipoles)]
	def generate_noise(self, batch_size, num_signals, num_nodes, num_dipoles=7498):
		return [torch.randn(batch_size, num_signals, num_dipoles)]


if __name__ == "__main__":
	z = torch.randn((16, 100, 7498))#.cuda()
	G = ForwardModelEnabledG(44, 50)	
	# G.cuda()
	out = G(z)
	print(out.shape)
	# print(out)


