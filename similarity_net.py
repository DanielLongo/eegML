import torch
from torch import nn

def contrastive_loss(euclidean_distance, label, margin=2.0):
	# Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
		(label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
	return loss_contrastive

class SimilarityNet(nn.Module):
	#input shape (seq_len, batch, input_size)
	#output shape (seq_len, batch, input_size)
	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True):
		super(SimilarityNet, self).__init__()
		self.bidirectional = bidirectional
		self.num_layers= num_layers
		self.num_nodes = num_nodes
		self.d = d
		self.rnn1 = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=self.num_layers, bidirectional=self.bidirectional)
		self.rnn2 = nn.LSTM(input_size=self.d * (1 + (bidirectional*1)), hidden_size=self.num_nodes, num_layers=1, bidirectional=False)
		# (seq_len, batch, num_directions * hidden_size)
		# (num_layers * num_directions, batch, hidden_size)
	def forward(self, noisy_data, clean_data):
		noisy_data, (hn, cn) = self.rnn1(noisy_data)
		noisy_data, (hn, cn) = self.rnn2(noisy_data)
		clean_data, (hn, cn) = self.rnn1(clean_data)
		clean_data, (hn, cn) = self.rnn2(clean_data)
		euclidean_distance = torch.nn.functional.pairwise_distance(noisy_data, clean_data)
		return euclidean_distance
		

# if __name__ == "__main__":
