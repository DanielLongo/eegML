import torch
from torch import nn

class ConditionalGenerator(nn.Module):
	#input shape (seq_len, batch, input_size)
	#output shape (seq_len, batch, input_size)
	def __init__(self, num_nodes, d, y_input_size, num_layers=3, bidirectional=True):
		super(ConditionalGenerator, self).__init__()
		self.bidirectional = bidirectional
		self.num_layers= num_layers
		self.num_nodes = num_nodes
		self.d = d
		self.fc1 = nn.Linear(y_input_size, num_nodes)
		self.rnn1 = nn.LSTM(input_size=self.num_nodes*2, hidden_size=self.d, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.rnn2 = nn.LSTM(input_size=self.d * (1 + (bidirectional*1)), hidden_size=self.num_nodes, num_layers=1, bidirectional=False, batch_first=True)
		# (seq_len, batch, num_directions * hidden_size)
		# (num_layers * num_directions, batch, hidden_size)
	def forward(self, x, y):
		y = self.fc1(y)
		x = torch.cat((x,y), 2)
		out, _ = self.rnn1(x)
		out, _ = self.rnn2(out)
		return out

if __name__ == "__main__":
	g = ConditionalGenerator(5, 50, 40)
	x = torch.randn(10, 20, 5)
	z = torch.randn(10, 20, 40)
	print(g(x, z).shape)
