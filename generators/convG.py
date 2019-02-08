import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
	def __init__(self, img_shape, latent_dim):
		self.latent_dim = latent_dim
		self.img_size = img_shape[1]
		self.channels = img_shape[0]
		super(Generator,  self).__init__()

		self.init_size = self.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128*self.init_size**2))

		self.conv_blocks = nn.Sequential(
			nn.BatchNorm2d(128),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
			nn.Tanh()
		)

	def forward(self, z):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks(out)
		return img