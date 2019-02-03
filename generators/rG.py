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
		self.rnn1 = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.rnn2 = nn.LSTM(input_size=self.d * (1 + (bidirectional*1)), hidden_size=self.num_nodes, num_layers=1, bidirectional=False, batch_first=True)
		# (seq_len, batch, num_directions * hidden_size)
		# (num_layers * num_directions, batch, hidden_size)
	# def forward(self, x):
		# out, _ = self.rnn1(x)
		# out, _ = self.rnn2(out)
		# return out

	def forward(self, x, *args, continuous=False):
		self.rnn1.flatten_parameters()
		self.rnn2.flatten_parameters()
		if len(args) != 0:
			assert(len(args) == 2), "Invalid args need len 2: h_0 and c_0"
			out, (h_n, c_n) = self.rnn1(x, (args[0], args[1])) #h_0, c_0
			out, _ = self.rnn2(out)
		else:
			out, (h_n, c_n) = self.rnn1(x)
			out, _ = self.rnn2(out)
		if continuous:
			return out, (h_n, c_n)
		return (out * 40)

	def generate_noise(self, batch_size, num_signals, num_nodes):
		return [torch.randn(batch_size, num_signals, num_nodes)]

if __name__ == "__main__":
	g = RecurrentGenerator(44, 50)
	noise = g.generate_noise(4, 100, 44)[0]
	h_0, c_0 = torch.zeros((6, 4, 50)), torch.zeros((6, 4, 50))
	out, (h_n, c_n) = g(noise, h_0, c_0, continuous=True)
	print("out", out.shape)
	print("h_n", h_n.shape)
	print("c_n", c_n.shape)