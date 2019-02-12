import argparse
import os
import numpy as np
import math
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

sys.path.append("./generators")
sys.path.append("./discriminators")
from convG_eeg import ConvGenerator
from convD_eeg import ConvDiscriminator
from tfr_convD import ConvDiscriminatorTFR
from tfr_convG import ConvGeneratorTFR
from load_EEGs import EEGDataset
from utils import save_EEG_tfr

import numpy as np
import matplotlib.pyplot as plt

#### for tfr ####
import pywt
import pywt.data

# import eeghdf
import mne
import mne.io
import h5py
wtype = 'sym3'

########

os.makedirs('images', exist_ok=True)

n_epochs = 200
batch_size = 64
lr = 0.00005
lr = .0000005
b1 = .5
b2 = .999
n_cpu = 8
latent_dim = 100
img_size = 32
channels = 1
sample_interval = 40000

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = ConvGeneratorTFR()
discriminator = ConvDiscriminatorTFR()

generate_noise = generator.generate_noise

if cuda:
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()

# Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# Configure data loader
real_eegs = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=50000, num_channels=44, batch_size=batch_size, length=1000, delay=100000)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(n_epochs):
	real_eegs.shuffle()
	for i, imgs in enumerate(real_eegs):
		if (imgs.shape[0] != batch_size):
			continue

		# Adversarial ground truths
		valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)#.cuda()
		fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)#.cuda()

		# Configure input
		real_imgs = Variable(imgs.type(Tensor))

		# -----------------
		#  Train Generator
		# -----------------

		optimizer_G.zero_grad()

		# Sample noise as generator input
		z = Variable(generate_noise(batch_size))

		if cuda:
			z = z.cuda()

		# Generate a batch of images
		fake_ch5A, fake_ch5D = generator(z)# * 20

		# Loss measures generator's ability to fool the discriminator
		g_loss = adversarial_loss(discriminator(fake_ch5A, fake_ch5D), valid)

		g_loss.backward()
		optimizer_G.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------

		optimizer_D.zero_grad()

		# Measure discriminator's ability to classify real from generated samples
		real_ch5A, real_ch5D = pywt.dwt(real_imgs.cpu().numpy(), wtype)
		real_ch5A = torch.from_numpy(real_ch5A).cuda()
		real_ch5D = torch.from_numpy(real_ch5D).cuda() 
		real_loss = adversarial_loss(discriminator(real_ch5A, real_ch5D), valid)
		fake_loss = adversarial_loss(discriminator(fake_ch5A.detach(), fake_ch5D.detach()), fake)
		d_loss = (real_loss + fake_loss) / 2

		d_loss.backward()
		optimizer_D.step()

		print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, i, len(real_eegs),
															d_loss.item(), g_loss.item()))
	save_EEG_tfr(fake_ch5A.cpu().detach().numpy(), fake_ch5D.cpu().detach().numpy(), 44, 200, "./generated_eegs/generated-"+ str(epoch) + "-fake-conv-tfr")
	print("Save @ Epoch", epoch)
		# batches_done = epoch * len(dataloader) + i
		# if batches_done % sample_interval == 0:
			# save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
