import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvGenerator(nn.Module):
	def __init__(self, img_shape, latent_dim):
		self.latent_dim = latent_dim
		self.img_shape = img_shape
		self.img_size = img_shape[1]
		self.channels = img_shape[0]
		super(ConvGenerator,  self).__init__()

		self.init_size = self.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128*self.init_size**2))

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

		self.conv_blocks = nn.Sequential(
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
		# out = self.l1(z)
		# out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		out = self.conv_blocks(z) * 20
		return out

	def generate_noise(self, batch_size):
		return torch.randn(batch_size, 1, 123, 5)

if __name__ == "__main__":
	# z = torch.randn((16, 100, 44))#.cuda()
	G = ConvGenerator((1, 100, 44), 100)
	z = G.generate_noise(30)
	out = G(z)
	print(out.shape)