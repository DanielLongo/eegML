import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvGenerator(nn.Module):
	def __init__(self):
		self.channels = 1
		super(ConvGenerator,  self).__init__()


		# self.conv_blocks = nn.Sequential(
		# 	nn.BatchNorm2d(128),
		# 	nn.Upsample(scale_factor=2),
		# 	nn.Conv2d(128, 128, 3, stride=1, padding=1),
		# 	nn.BatchNorm2d(128, 0.8),
		# 	nn.LeakyReLU(0.2, inplace=True),
		# 	nn.Upsample(scale_factor=2),
		# 	nn.Conv2d(128, 64, 3, stride=1, padding=1),
		# 	nn.BatchNorm2d(64, 0.8),
		# 	nn.LeakyReLU(0.2, inplace=True),
		# 	nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
		# 	nn.Tanh()
		# )

		self.conv_blocks_ch5A = nn.Sequential(
			nn.ConvTranspose2d(1, 4, [2,3], stride=[1,1], padding=0),
			nn.BatchNorm2d(4, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(4, 16, [4,4], stride=[2,1], padding=0),
			nn.BatchNorm2d(16, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(16, 8, [3,3], stride=[2,2], padding=0),
			nn.BatchNorm2d(8, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(8, self.channels, [4,4], stride=[2,2], padding=0),
			nn.BatchNorm2d(self.channels, 0.8),
			nn.Tanh()
		)

		self.conv_blocks_ch5D = nn.Sequential(
			nn.ConvTranspose2d(1, 4, [2,3], stride=[1,1], padding=0),
			nn.BatchNorm2d(4, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(4, 16, [4,4], stride=[2,1], padding=0),
			nn.BatchNorm2d(16, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(16, 8, [3,3], stride=[2,2], padding=0),
			nn.BatchNorm2d(8, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(8, self.channels, [4,4], stride=[2,2], padding=0),
			nn.BatchNorm2d(self.channels, 0.8),
			nn.Tanh()
		)

	def forward(self, z):
		#TODO: FIX SHAPES DURING CONV
		ch5A = self.conv_blocks_ch5A(z)[:,:, :1002, :]
		ch5D = self.conv_blocks_ch5D(z)[:,:, :1002, :]
		return ch5A, ch5D

	def generate_noise(self, batch_size):
		return torch.randn(batch_size, 1, 123, 5)

if __name__ == "__main__":
	# z = torch.randn((16, 100, 44))#.cuda()
	G = ConvGenerator()
	z = G.generate_noise(30)
	ch5A, ch5D = G(z)
	print(ch5A.shape)
	print(ch5D.shape)