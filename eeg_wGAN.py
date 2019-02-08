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
from load_EEGs import EEGDataset

os.makedirs('images', exist_ok=True)

n_epochs = 200
batch_size = 64
lr = 0.00005
b1 = .5
b2 = .999
n_cpu = 8
latent_dim = 100
img_size = 32
channels = 1
sample_interval = 400

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
generator = ConvGenerator(img_shape, latent_dim)
discriminator = ConvDiscriminator(img_shape)

generate_noise = generator.generate_noise

if cuda:
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()

# Initialize weights
def compute_gradient_penalty(D, real_samples, fake_samples):
	"""Calculates the gradient penalty loss for WGAN GP"""
	# Random weight term for interpolation between real and fake samples
	alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
	# Get random interpolation between real and fake samples
	interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
	d_interpolates = D(interpolates)
	fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
	# Get gradient w.r.t. interpolates
	gradients = autograd.grad(
		outputs=d_interpolates,
		inputs=interpolates,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# Configure data loader
real_eegs = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=5000, num_channels=44, batch_size=batch_size, length=1004, delay=100000)

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

		# Adversarial ground truths
		valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

		# Configure input
		real_imgs = Variable(imgs.type(Tensor))

		# -----------------
		#  Train Discriminator
		# -----------------

		optimizer_D.zero_grad()

		# Sample noise as generator input
		z = Variable(generate_noise(batch_size))

		# Generate a batch of images
		gen_imgs = generator(z)

		# Loss measures generator's ability to fool the discriminator
		gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
		real_validity = discriminator(real_imgs)
		fake_validity = discriminator(gen_imgs)
		d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
		#g_loss = adversarial_loss(discriminator(gen_imgs), valid)

		d_loss.backward()
		optimizer_D.step()

		# ---------------------
		#  Train Generator
		# ---------------------

		optimizer_G.zero_grad()

		# Measure discriminator's ability to classify real from generated samples
		fake_imgs = generator(z)

		fake_validity = discriminator(fake_imgs)
		g_loss = -torch.mean(fake_validity)

		g_loss.backward()
		optimizer_G.step()

		print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader),
															d_loss.item(), g_loss.item()))

		# batches_done = epoch * len(dataloader) + i
		# if batches_done % sample_interval == 0:
			# save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
