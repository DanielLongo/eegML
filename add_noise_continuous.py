import random
import torch
import sys

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("./generators/")
from rDiscriminator import RecurrentDiscriminator
from forward_model_enabled_G import ForwardModelEnabledG
from conditional_generator import ConditionalGenerator
from c_forward_model import cGForwardModel
from rG import RecurrentGenerator
from load_EEGs import EEGDataset
from utils import save_EEG

USE_CUDA = True
ITERS = 30
CRITIC_ITERS = 3 #3
BATCH_SIZE = 8
LAMBDA = 10 # Gradient penalty lambda hyperparameter
NUM_NODES = 44
LENGTH = 1000
H_DIM = 20

real_eegs = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=256*2, num_channels=NUM_NODES, batch_size=BATCH_SIZE)


one = torch.ones([])
mone = one * -1

## Pick Of Discriminators 
netD = RecurrentDiscriminator(num_nodes=NUM_NODES, d=H_DIM)

## Pick Of Generators
netG = RecurrentGenerator(num_nodes=NUM_NODES, d=H_DIM)
# netG = ForwardModelEnabledG(44, 50)
# netG = ConditionalGenerator(num_nodes=44, d=64, y_input_size=20)
# netG = cGForwardModel(num_nodes=44, d=64, y_input_size=20)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

if USE_CUDA:
	one = one.cuda()
	mone = mone.cuda()
	netD.cuda()
	netG.cuda()

def calc_gradient_penalty(netD, real_data, fake_data):
	alpha = torch.rand(BATCH_SIZE, 1, 1)
	alpha = alpha.expand(real_data.size())
	alpha = alpha.cuda() if USE_CUDA else alpha

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	if USE_CUDA:
		interpolates = interpolates.cuda()
	interpolates = autograd.Variable(interpolates, requires_grad=True)

	disc_interpolates = netD(interpolates)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).cuda() if USE_CUDA else torch.ones(
								  disc_interpolates.size()),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty

def main():
	for iteration in range(ITERS):
		h_n, c_n = torch.zeros((6, BATCH_SIZE, H_DIM)), torch.zeros((6, BATCH_SIZE, H_DIM))
		if USE_CUDA:
			h_n, c_n = h_n.cuda(), c_n.cuda()
		real_eegs.shuffle()
		for i in range(len(real_eegs)):
			if (real_eegs[i].shape[0] != BATCH_SIZE):
				continue
			############################
			# (1) Update D network
			###########################
			for p in netD.parameters():  # reset requires_grad
				p.requires_grad = True  # they are set to False below in netG update:

			# for p in netG.parameters():
			# 	p.requires_grad = False # to avoid computation

			for iter_d in range(CRITIC_ITERS):
				real = real_eegs[random.randint(0,len(real_eegs)-2)]

				if USE_CUDA:
					real = real.cuda()
				real_v = autograd.Variable(real)

				netD.zero_grad()

				# train with real
				D_real = netD(real_v)
				D_real = D_real.mean()
				D_real.backward(mone)

				# train with fake
				noise = netG.generate_noise(BATCH_SIZE, LENGTH, NUM_NODES)

				if USE_CUDA:
					noise = [x.cuda() for x in noise]

				noise_v = [autograd.Variable(x) for x in noise]

				if (len(noise) == 1): #not conditional
					h_0, c_0 = h_n, c_n
					fake, (h_n, c_n) = netG(noise_v[0], h_0, c_0, continuous=True)
					# fake, (h_n, c_n) = netG(noise_v[0], h_n, c_n, continuous=True)
					fake = autograd.Variable(fake.data)
				# if (len(noise) == 2): #conditional with 1 Y
				# 	fake, (h_n, c_n) = netG(noise_v[0], noise_v[1], continuous=True).data
				# 	fake = autograd.Variable(fake)
				# if (len(noise) == 3): #conditional with 2 Ys
				# 	fake, (h_n, c_n) = netG(noise_v[0], noise_v[1], noise_v[2] continuous=True).data
				# 	fake = autograd.Variable(fake)

				D_fake = netD(fake, h_0, c_0)
				# D_fake = netD(fake)
				D_fake = D_fake.mean()
				# D_fake.backward(one)
				D_fake.backward(one, retain_graph=True)


				# train with gradient penalty
				gradient_penalty = calc_gradient_penalty(netD, real_v.data, fake.data)
				#gradient_penalty.backward()

				D_cost = D_fake - D_real + gradient_penalty
				Wasserstein_D = D_real - D_fake
				optimizerD.step()

		############################
		# (2) Update G network
		###########################
			for p in netD.parameters():
				p.requires_grad = False  # to avoid computation

			# for p in netG.parameters():
			# 	p.requires_grad = True

			netG.zero_grad()

			noise = netG.generate_noise(BATCH_SIZE, LENGTH, NUM_NODES)

			if USE_CUDA:
				noise = [x.cuda() for x in noise]

			noise_v = [autograd.Variable(x) for x in noise]

			if (len(noise) == 1): #not conditional
				h_0, c_0 = h_n, c_n
				fake, (h_n, c_n) = netG(noise_v[0], h_n, c_n, continuous=True)
				# fake, (h_n, c_n) = netG(noise_v[0], h_n, c_n, continuous=True)
				fake = autograd.Variable(fake.data)
			# if (len(noise) == 2): #conditional with 1 Y
			# 	fake = autograd.Variable(netG(noise_v[0], noise_v[1], continuous=True).data)
			# if (len(noise) == 3): #conditional with 2 Ys
			# 	fake = autograd.Variable(netG(noise_v[0], noise_v[1], noise_v[2], continuous=True).data)

			G = netD(fake, h_0, c_0)
			# G = netD(fake)
			G = G.mean()
			# G.requires_grad = True
			# G.backward(mone)
			G.backward(mone, retain_graph=True)
			G_cost = -G
			optimizerG.step()

		print("Epoch", iteration)
		print("G_cost" , G_cost)
		print("D_cost", D_cost)

		if (iteration % 1 == 0):
			save_EEG(fake.cpu().detach().numpy(), NUM_NODES, 200, "./generated_eegs/generated-"+ str(iteration) + "-fake-rG-cont")
			# save_EEG(real.cpu().detach().numpy(), NUM_NODES, 200, "./generated_eegs/generated-"+ str(iteration) + "-real-rGF-shuffle")
if __name__ == "__main__":
	main()
