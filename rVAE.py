#adapted from https://github.com/caogang/wgan-gp


import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
	gpu = 0

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many Decoder iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())

# ==================Definition Start======================
class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

	#TODO
	#need to workout shape of output
	def __init__(self, time, num_nodes, y_shape, d):
		super(RecurrentGenerator, self).__init__()
		self.time = time
		self.num_nodes = num_nodes
		self.d = d
		self.rnn1 = nn.LSTM(input_size=self.num_nodes, hidden_size=self.d, num_layers=3, bidirectional=True, nonlinearity="relu")
		self.rnn2 = nn.LSTM(input_size=(d + y_shape[1]), hidden_size=self.d, num_layers=3, bidirectional=True, nonlinearity="relu") 
		self.fc1 = nn.Sequential(
			 nn.Linear(self.d, self.num_nodes),
			 # nn.tanh()
			 #TODO Find an activation that fits eeg generation
		)

	def forward(self, x, y):
		# x = (batch_size, time, num_sensors)
		# y = (batch_size, num_features)
		# num_features of y must match hidden d
		assert(y.shape[1] == self.d)
		out = self.rnn1(x)
		out = torch.cat(out,y)
		out = self.rnn2(out)
		out = self.fc1(out)
		return out

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

	def forward(self, x):
		############### need to use forward model
		return x

# Dataset iterator
# train_gen, dev_gen, test_gen =
def inf_train_gen():
	while True:
		for images,targets in train_gen():
			yield images

def calc_gradient_penalty(netE, real_data, fake_data):
	#print real_data.size()
	alpha = torch.rand(BATCH_SIZE, 1)
	alpha = alpha.expand(real_data.size())
	alpha = alpha.cuda(gpu) if use_cuda else alpha

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	if use_cuda:
		interpolates = interpolates.cuda(gpu)
	interpolates = autograd.Variable(interpolates, requires_grad=True)

	disc_interpolates = netE(interpolates)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
								  disc_interpolates.size()),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty

# ==================Definition End======================

netD = Decoder()
netE = Encoder()
print netD
print netE

if use_cuda:
	netE = netE.cuda(gpu)
	netD = netD.cuda(gpu)

optimizerD = optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
	one = one.cuda(gpu)
	mone = mone.cuda(gpu)

data = inf_train_gen()
for p in netE.parameters():
	p.requires_grad = False  # to avoid computation

for iteration in xrange(ITERS):
	start_time = time.time()
	############################
	# Update Decoder network 
	# No need to update physics based encoder network
	###########################
		_data = data.next()
		real_data = torch.Tensor(_data)
		if use_cuda:
			real_data = real_data.cuda(gpu)
		real_data_v = autograd.Variable(real_data)

		netE.zero_grad()


		localized_activity = autograd.Variable(netD(real_data))
		E_fake = netE(localized_activity)
		E_fake = E_fake.mean()
		E_fake.backward(mone)
		E_fake = -E_fake

		# train with gradient penalty
		gradient_penalty = calc_gradient_penalty(netE, real_data_v.data, fake.data)
		gradient_penalty.backward()



