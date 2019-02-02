import torch
from torch import nn

class RecurrentDiscriminator(nn.Module):
	#input shape (seq_len, batch, input_size)
	#ourput shape (batch, 1)
	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True):
		super(RecurrentDiscriminator, self).__init__()
		self.bidirectional = bidirectional
		self.num_layers= num_layers
		self.num_nodes = num_nodes
		self.d = d
		self.rnn1 = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.rnn2 = nn.LSTM(input_size=self.d * (1 + (bidirectional*1)), hidden_size=self.num_nodes, num_layers=1, bidirectional=False, batch_first=True)
		self.fc1 = nn.Sequential(
			nn.Linear(self.num_nodes * 1, self.num_nodes//2),
			nn.ReLU()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(self.num_nodes//2, 1),
			nn.Sigmoid()
		)


	# def forward(self, x):
	# 	output, (hn, cn) = self.rnn1(x)
	# 	out = output
	# 	output, (hn, cn) = self.rnn2(out) 
	# 	out = cn #TODO WHICH ONE
	# 	out = out.view(-1, self.num_nodes * 1)
	# 	out = self.fc1(out)
	# 	out = self.fc2(out)
	# 	return out

	def forward(self, x, *args, return_states=False):
		self.rnn1.flatten_parameters()
		self.rnn2.flatten_parameters()
		if len(args) != 0:
			assert(len(args) == 2), "Invalid args need len 2: h_0 and c_0"
			out, (h_n, c_n) = self.rnn1(x, (args[0], args[1])) #h_0, c_0
			# out, _ = self.rnn1(x, (args[0], args[1])) #h_0, c_0
		else:
			out, (h_n, c_n) = self.rnn1(x)
			# out, _ = self.rnn1(x)

		# out, (h_n, c_n) = self.rnn2(out)
		_, (_,out) = self.rnn2(out)
		# out = h_n #TODO WHICH ONE
		out = out.view(-1, self.num_nodes * 1)
		out = self.fc1(out)
		out = self.fc2(out)
		if (return_states):
			return out, (h_n, c_n)
		return out

if __name__ == "__main__":
	d = RecurrentDiscriminator(44, 50)
	x = torch.zeros(64, 100, 44)
	h_0 = torch.zeros(6, 64, 50)
	c_0 = torch.zeros(6, 64, 50)
	pred = d(x, h_0, c_0)
	print(pred.shape)
	pred = d(x)
	print(pred.shape)