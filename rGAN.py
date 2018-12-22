import torch
from torch import nn

class RecurrentGenerator(nn.Module):
	#TODO
	#work out shapes of input !
	def __init__(self, time, num_nodes, d):
		super(RecurrentGenerator, self).__init__()
		self.time = time
		self.num_nodes = num_nodes
		self.d = d
		self.rnn = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=3, bidirectional=True, nonlinearity="relu")
		self.fc1 = nn.Sequential(
			 nn.Linear(self.d, self.num_nodes),
			 # nn.tanh()
			 #TODO Find an activation that fits eeg generation
		)

	def forward(self, x):
		out = self.rnn(x)
		out = self.fc1(out)
		return out

def RecurrentDiscriminator(nn.Module):
	def __init__(self, time, num_nodes, d):
		super(RecurrentDiscriminator, self).__init__()
		self.time = time
		self.num_nodes = num_nodes
		self.d = d
		self.rnn = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=3, bidirectional=True, nonlinearity="relu")
		self.fc1 = nn.Sequential(
			nn.Linear(self.d * self.time, self.d),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear((self.d), 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		out = self.rnn(x)
		out = out.view(-1, self.num_nodes, self.d*self.time)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

