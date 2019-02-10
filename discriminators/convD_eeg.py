import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class ConvDiscriminator(nn.Module):
	def __init__(self, img_shape):
		self.img_size = img_shape[1]
		self.channels = img_shape[0]
		super(ConvDiscriminator, self).__init__()

		def discriminator_block(in_filters, out_filters, bn=True):
			block = [   nn.Conv2d(in_filters, out_filters, [3,5], [2,1], padding=0),
						nn.LeakyReLU(0.2, inplace=True),
						nn.Dropout2d(0.25)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters, 0.8))
			return block

		self.model = nn.Sequential(
			*discriminator_block(self.channels, 16, bn=False),
			*discriminator_block(16, 32),
			*discriminator_block(32, 64),

			*discriminator_block(64, 32),
			*discriminator_block(32, 16),
			*discriminator_block(16, 8),

			*discriminator_block(8, 4),
		)

		# The height and width of downsampled image
		self.fc1 = nn.Sequential( nn.Linear(4*6*16, 16),
										nn.LeakyReLU(.2, inplace=True))
		self.fc2 = nn.Sequential( nn.Linear(16, 1),
										nn.Sigmoid())

	def forward(self, img, matching=False):
		if (len(img.shape) == 3):
			img = img.view(img.shape[0], 1, img.shape[1], img.shape[2])
		out = self.model(img)
		out = out.view(out.shape[0], -1)
		out = self.fc1(out)
		final = self.fc2(out)
		if matching:
			return out, final
		return final
		
# class ConvDiscriminator(nn.Module):
# 	def __init__(self, img_shape):
# 		super(ConvDiscriminator, self).__init__()

# 		self.model = nn.Sequential(
# 			nn.Linear(int(np.prod(img_shape)), 512),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Linear(512, 256),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Linear(256, 1)
# 		)

# 	def forward(self, img):
# 		img_flat = img.view(img.shape[0], -1)
# 		validity = self.model(img_flat)
# 		return validity

if __name__ == "__main__":
	# x = torch.ones(64, 100, 42)
	x = torch.ones(64, 1, 32, 32)
	d = ConvDiscriminator((1,32,32))
	z = d(x)
	# print("Z",z)
	print("Z shape", z.shape)
