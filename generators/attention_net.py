import torch
from torch import nn
#from https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/patterns/attention.html
class Attention(nn.Module):
	def __init__(self, attention_size):
		super(Attention, self).__init__()
		self.attention = new_parameter(attention_size, 1)

	def forward(self, x_in):
		# after this, we have (batch, dim1) with a diff weight per each cell
		attention_score = torch.matmul(x_in, self.attention).squeeze()
		attention_score = F.softmax(attention_score).view(x_in.size(0), x_in.size(1), 1)
		scored_x = x_in * attention_score

		# now, sum across dim 1 to get the expected feature vector
		condensed_x = torch.sum(scored_x, dim=1)

		return condensed_x

if __name__ == "__main__":
	attn = Attention(100)
	x = Variable(torch.randn(16,30,100))
	attn(x).size() == (16,100)