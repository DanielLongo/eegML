import torch
from torch import nn
import torch.nn.functional as F
#  # super(DecoderRNN, self).__init__()
#  #        self.hidden_size = hidden_size

#  #        self.embedding = nn.Embedding(output_size, hidden_size)
#  #        self.gru = nn.GRU(hidden_size, hidden_size)
#  #        self.out = nn.Linear(hidden_size, output_size)
#  #        self.softmax = nn.LogSoftmax(dim=1)
# class RecurrentDiscriminator(nn.Module):
# 	#input shape (seq_len, batch, input_size)
# 	#ourput shape (batch, 1)
# 	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True):
# 		super(RecurrentDiscriminator, self).__init__()
# 		self.bidirectional = bidirectional
# 		self.num_layers= num_layers
# 		self.num_nodes = num_nodes
# 		self.d = d
# 		self.mid_d = d//2
# 		self.rnn1 = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
# 		self.embedding = nn.Embedding(1, self.d * (1 + (bidirectional*1)))
# 		self.embedding = nn.Embedding(1, self.d)
# 		self.rnn2 = nn.LSTM(self.d * (1 + (bidirectional*1)),1, num_layers=1, bidirectional=False, batch_first=True)
# 		self.softmax = nn.LogSoftmax(dim=1)
# 		self.out = nn.Linear(self.d, 1)
# 		# self.fc1 = nn.Sequential(
# 		# 	nn.Linear(self.num_nodes * 1, self.num_nodes//2),
# 		# 	nn.ReLU()
# 		# )
# 		# self.fc2 = nn.Sequential(
# 		# 	nn.Linear(self.num_nodes//2, 1),
# 		# 	nn.Sigmoid()
# 		# )


# 	# def forward(self, x):
# 	# 	output, (hn, cn) = self.rnn1(x)
# 	# 	out = output
# 	# 	output, (hn, cn) = self.rnn2(out) 
# 	# 	out = cn #TODO WHICH ONE
# 	# 	out = out.view(-1, self.num_nodes * 1)
# 	# 	out = self.fc1(out)
# 	# 	out = self.fc2(out)
# 	# 	return out

# 	def forward(self, x, *args, return_states=False, matching=False):
# 		self.rnn1.flatten_parameters()
# 		self.rnn2.flatten_parameters()
# 		if len(args) != 0:
# 			assert(len(args) == 2), "Invalid args need len 2: h_0 and c_0"
# 			out, (h_n, c_n) = self.rnn1(x, (args[0], args[1])) #h_0, c_0
# 			# out, _ = self.rnn1(x, (args[0], args[1])) #h_0, c_0
# 		else:
# 			out, (h_n, c_n) = self.rnn1(x)
# 			# out, _ = self.rnn1(x)

# 		# out, (h_n, c_n) = self.rnn2(out)
# 		print("after rnn 1", out.shape)
# 		out = out.long()
# 		a = out
# 		# out = self.embedding(out).view(-1,self.d,self.mid_d)
# 		c_n = c_n.long()
# 		out = self.embedding(c_n).view(1,1,-1)
# 		out = F.relu(out)
# 		print("after embed", out.shape)
# 		output, (_, out) = self.rnn2(out)
# 		print("after rnn 2", output.shape)
# 		# out = F.sigmoid(out)
# 		print("output", output.shape)
# 		output = self.softmax(self.out(output[0]))
# 		return output

class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=1000):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input):#, hidden, encoder_outputs):
		input = input.long()
		embedded = self.embedding(input)#.view(1, 1, -1)
		# embedded = self.dropout(embedded)
		print("embedded", embedded.shape)

		# print("attn", attn_weights.shape)
		# attn_applied = torch.bmm(attn_weights.unsqueeze(0),
								 # encoder_outputs.unsqueeze(0))

		# output = torch.cat((embedded[0], attn_applied[0]), 1)
		# output = self.attn_combine(output).unsqueeze(0)

		output = embedded
		output, hidden = self.gru(output)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)
if __name__ == "__main__":
	# d = RecurrentDiscriminator(44, 50)
	d = AttnDecoderRNN(44, 1)
	x = torch.zeros(64, 100, 44)
	# h_0 = torch.zeros(6, 64, 50)
	# c_0 = torch.zeros(6, 64, 50)
	# pred = d(x, h_0, c_0)
	# print(pred.shape)
	pred = d(x)
	# print(pred.shape)