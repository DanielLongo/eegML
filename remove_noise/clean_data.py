import torch
from torch import nn

class CleanData(nn.Module):
	#input shape (seq_len, batch, input_size)
	#out shape (seq_len, batch, input_size)
	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True):
		super(CleanData, self).__init__()
		self.bidirectional = bidirectional
		self.num_layers= num_layers
		self.num_nodes = num_nodes
		self.d = d
		self.rnn1 = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=self.num_layers, bidirectional=self.bidirectional)
		self.rnn2 = nn.LSTM(input_size=self.d * (1 + (bidirectional*1)), hidden_size=self.num_nodes, num_layers=1, bidirectional=False)
		# (seq_len, batch, num_directions * hidden_size)
		# (num_layers * num_directions, batch, hidden_size)
	def forward(self, x):
		out, (hn, cn) = self.rnn1(x)
		out, (hn, cn) = self.rnn2(out)
		return out 
		

if __name__ == "__main__":
	BATCH_SIZE = 4
	NUM_NODES = 32
	TIME = 10
	noisy_data = torch.randn(BATCH_SIZE, TIME, NUM_NODES)
	cleaner = CleanData(NUM_NODES, 50)
	cleaned = cleaner(noisy_data)
	print("cleaned", cleaned.shape)