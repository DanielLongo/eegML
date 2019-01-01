import torch
from torch import nn

class RecurrentGenerator(nn.Module):
	#input shape (seq_len, batch, input_size)
	#output shape (seq_len, batch, input_size)
	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True):
		super(RecurrentGenerator, self).__init__()
		self.bidirectional = bidirectional
		self.num_layers= num_layers
		self.num_nodes = num_nodes
		self.d = d
		self.rnn1 = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=self.num_layers, bidirectional=self.bidirectional)
		self.rnn2 = nn.LSTM(input_size=self.d * (1 + (bidirectional*1)), hidden_size=self.num_nodes, num_layers=1, bidirectional=False)
		# (seq_len, batch, num_directions * hidden_size)
		# (num_layers * num_directions, batch, hidden_size)
	def forward(self, x):
		out, _ = self.rnn1(x)
		out, _ = self.rnn2(out)
		return out

class RecurrentDiscriminator(nn.Module):
	#input shape (seq_len, batch, input_size)
	#ourput shape (batch, 1)
	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True):
		super(RecurrentDiscriminator, self).__init__()
		self.bidirectional = bidirectional
		self.num_layers= num_layers
		self.num_nodes = num_nodes
		self.d = d
		self.rnn1 = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=self.num_layers, bidirectional=self.bidirectional)
		self.rnn2 = nn.LSTM(input_size=self.d * (1 + (bidirectional*1)), hidden_size=self.num_nodes, num_layers=1, bidirectional=False)
		self.fc1 = nn.Sequential(
			nn.Linear(self.num_nodes * 1, self.num_nodes//2),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(self.num_nodes//2, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		output, (hn, cn) = self.rnn1(x)
		out = output
		output, (hn, cn) = self.rnn2(out) 
		out = cn #TODO WHICH ONE
		out = out.view(-1, self.num_nodes * 1)
		out = self.fc1(out)
		out = self.fc2(out)
		return out


