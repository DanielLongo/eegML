import torch
import torch.nn as nn
import torch.nn.functional as F
from rDiscriminator import RecurrentDiscriminator

class rDMinBatch(nn.Module):

	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True):
		"""
		Minibatch discrimination: learn a tensor to encode side information
		from other examples in the same minibatch.
		"""
		super(rDMinBatch, self).__init__()
		self.d = d
		self.rnn = RecurrentDiscriminator(num_nodes, d, num_layers=num_layers, bidirectional=bidirectional)
		self.featmap_dim = num_nodes//2
		T_ten_init = torch.randn(self.featmap_dim, d) * 0.1
		self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)
		self.fc = nn.Linear(self.featmap_dim + self.d, 1)

	def forward(self, x, matching=False):
		"""
		Architecture is similar to DCGANs
		Add minibatch discrimination => Improved GAN.
		"""
		x, _ = self.rnn(x, matching=True)
		# print("x", x.shape)
		x = x.view(-1, self.featmap_dim)
		matching_array = x

		# #### Minibatch Discrimination ###
		T_tensor = self.T_tensor

		Ms = x.mm(T_tensor)
		# print(Ms.shape)
		Ms = Ms.view(-1, self.d, 1)

		out_tensor = []
		for i in range(Ms.size()[0]):

			out_i = None
			for j in range(Ms.size()[0]):
				o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
				o_i = torch.exp(-o_i)
				if out_i is None:
					out_i = o_i
				else:
					out_i = out_i + o_i

			out_tensor.append(out_i)

		out_T = torch.cat(tuple(out_tensor)).view(Ms.size()[0], self.d)
		x = torch.cat((x, out_T), 1)
		# #### Minibatch Discrimination ###
		# print('x', x.shape)
		x = torch.sigmoid(self.fc(x))
		if matching:
			return matching_array, x
		return x

if __name__ == "__main__":
	x = torch.ones(64, 100, 42)
	d = rDMinBatch(42, 50)
	z = d(x)
	# print("z", z.shape)
