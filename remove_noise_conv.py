import torch
from torch import nn
import sys
sys.path.append("./generators/")
sys.path.append("./discriminators/")


class ConvRemoveNoise(nn.Module):
	#input shape (seq_len, batch, input_size)
	#output shape (seq_len, batch, input_size)
	def __init__(self, num_nodes, d, num_layers=3, bidirectional=True):
		super(ConvRemoveNoise, self).__init__()
		self.channels_out = 1
		self.channels_h = 4

		self.conv_blocks_decode = nn.Sequential(
			nn.ConvTranspose2d(self.channels_h, 4, [2,3], stride=[1,1], padding=0),
			nn.BatchNorm2d(4, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(4, 16, [4,4], stride=[2,1], padding=0),
			nn.BatchNorm2d(16, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(16, 8, [3,3], stride=[2,2], padding=0),
			nn.BatchNorm2d(8, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(8, self.channels_out, [4,4], stride=[2,2], padding=0),
			nn.BatchNorm2d(self.channels_out, 0.8),
			nn.Tanh()
		)

		self.conv_blocks_encode  = nn.Sequential (
			nn.Conv2d(self.channels_out, 4, [3,4], stride=[2,3], padding=0),
			nn.BatchNorm2d(4, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(4, 16, [6,4], stride=[2,1], padding=0),
			nn.BatchNorm2d(16, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(16, self.channels_h, [3,3], stride=[2,2], padding=0),
			nn.BatchNorm2d(self.channels_h, 0.8),
			nn.Tanh()
		)

	def forward(self, x):
		if (len(x.shape) == 3):
			x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
		encoded = self.conv_blocks_encode(x)
		# print("encoded", encoded.shape)
		decoded = self.conv_blocks_decode(encoded)
		return decoded

if __name__ == "__main__":
	g = ConvRemoveNoise(44, 50)
	sample = torch.randn(1, 1, 1004, 44)
	out = g(sample)
	print("out", out.shape)
