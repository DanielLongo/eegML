import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class ConvDiscriminatorTFR(nn.Module):
	def __init__(self):
		super(ConvDiscriminatorTFR, self).__init__()

		def discriminator_block(in_filters, out_filters, bn=True):
			block = [   nn.Conv2d(in_filters, out_filters, [3,3], [2,1], padding=0),
						nn.LeakyReLU(0.2, inplace=True),
						nn.Dropout2d(0.25)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters, 0.8))
			return block

		self.model_ch5A = nn.Sequential(
			*discriminator_block(1, 16, bn=False),
			*discriminator_block(16, 32),
			*discriminator_block(32, 64),

			*discriminator_block(64, 32),
			*discriminator_block(32, 32),
			*discriminator_block(32, 32),

			*discriminator_block(32, 32),
			*discriminator_block(32, 8),
		)

		self.model_ch5D = nn.Sequential(
			*discriminator_block(1, 16, bn=False),
			*discriminator_block(16, 32),
			*discriminator_block(32, 64),

			*discriminator_block(64, 32),
			*discriminator_block(32, 32),
			*discriminator_block(32, 32),

			*discriminator_block(32, 32),
			*discriminator_block(32, 8),
		)

		# The height and width of downsampled image
		self.fc1_ch5A = nn.Sequential( nn.Linear(2 * 8 * 8, 128),
										nn.LeakyReLU(.2, inplace=True))
		self.fc1_ch5D = nn.Sequential( nn.Linear(2 * 8 * 8, 128),
										nn.LeakyReLU(.2, inplace=True))
		self.fc2 = nn.Sequential( nn.Linear(256, 64),
										nn.LeakyReLU(.2, inplace=True))
		self.fc3 = nn.Sequential( nn.Linear(64, 1),
			nn.Sigmoid())

	def forward(self, ch5A, ch5D, matching=False):
		if (len(ch5A.shape) == 3):
			ch5A = ch5A.view(ch5A.shape[0], 1, ch5A.shape[1], ch5A.shape[2])
		if (len(ch5D.shape) == 3):
			ch5D = ch5D.view(ch5D.shape[0], 1, ch5D.shape[1], ch5D.shape[2])

		# print(ch5A.shape)
		out_ch5A = self.model_ch5A(ch5A)
		# print(out_ch5A.shape)
		out_ch5A = out_ch5A.view(out_ch5A.shape[0], -1)
		out_ch5A = self.fc1_ch5A(out_ch5A)
		out_ch5D = self.model_ch5D(ch5D)
		# print(out_ch5D.shape)
		out_ch5D = out_ch5D.view(out_ch5D.shape[0], -1)
		out_ch5D = self.fc1_ch5D(out_ch5D)
		out = torch.cat((out_ch5A, out_ch5D), dim=1)
		out = self.fc2(out)
		out = self.fc3(out)
		return out
		
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
	xA = torch.ones(64, 1, 1000, 24)
	xD = torch.ones(64, 1, 1000, 24)
	# x = torch.ones(64, 1, 32, 32)
	d = ConvDiscriminatorTFR()
	z = d(xA,xD)
	# print("Z",z)
	print("Z shape", z.shape)
