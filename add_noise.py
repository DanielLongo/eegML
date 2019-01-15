import torch
from rGAN import RecurrentGenerator, RecurrentDiscriminator
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_EEGs import EEGDataset

use_cuda = False
ITERS = 10
CRITIC_ITERS = 3
BATCH_SIZE = 4
LAMBDA = 10 # Gradient penalty lambda hyperparameter
NUM_NODES= 44

noisy_eegs = torch.randn(16, BATCH_SIZE, 100, NUM_NODES) #None, batch_size, time, num_nodes
# noisy_eegs = EEGDataset("./the_server/Users/DanielLongo/server/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=10, num_channels=NUM_NODES, batch_size=BATCH_SIZE)
clean_eegs = torch.randn(16, BATCH_SIZE, 100, NUM_NODES)
# clean_eegs = EEGDataset("./the_server/Users/DanielLongo/server/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=10, num_channels=NUM_NODES, batch_size=BATCH_SIZE)


one = torch.FloatTensor([1])
# one = [torch.FloatTensor(1).cuda() for _ in range(1)]
mone = one * -1
if use_cuda:
	one = one.cuda()
	mone = mone.cuda()

netD = RecurrentDiscriminator(NUM_NODES, 64)
netG = RecurrentGenerator(num_nodes=NUM_NODES, d=50)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

def calc_gradient_penalty(netD, real_data, fake_data):
	# print(real_data.size())
	alpha = torch.rand(BATCH_SIZE, 1, 1)
	alpha = alpha.expand(real_data.size())
	# print("alpha", alpha.size())
	alpha = alpha.cuda() if use_cuda else alpha

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	if use_cuda:
		interpolates = interpolates.cuda()
	interpolates = autograd.Variable(interpolates, requires_grad=True)

	disc_interpolates = netD(interpolates)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
								  disc_interpolates.size()),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty

def main():
	for iteration in range(ITERS):
		# for i in range(noisy_eegs.shape[0]):
		for i in range(len(noisy_eegs)):
			############################
			# (1) Update D network
			###########################
			for p in netD.parameters():  # reset requires_grad
				p.requires_grad = True  # they are set to False below in netG update
			# for p in netG.parameters():
			# 	p.requires_grad = False

			for iter_d in range(CRITIC_ITERS):
				real_data = noisy_eegs[i]
				real_data_v = autograd.Variable(real_data)
				if use_cuda:
					real_data = real_data.cuda()

				netD.zero_grad()

				# train with real
			D_real = netD(real_data_v)
			D_real = D_real.mean()
			# print D_real
			# D_real.backward(mone)
			D_real.backward()

			# train with fake
			fake_data = clean_eegs[i]
			if use_cuda:
				fake_data = fake_data.cuda()
			fake_datav = autograd.Variable(fake_data, volatile=True)  # totally freeze netG
			fake_data.requires_grad = False
			# fake_data.requires_grad = True
			# fake_datav = fake_data
			fake = autograd.Variable(netG(fake_datav).data)
			inputv = fake
			# inputv = fake_data
			D_fake = netD(inputv)
			D_fake = D_fake.mean()
			# D_fake.backward(one)
			D_fake.backward()


			# train with gradient penalty
			gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
			gradient_penalty.backward()

			D_cost = D_fake - D_real #+ gradient_penalty
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

		fake_data = clean_eegs[i]
		if use_cuda:
			fake_data = fake_data.cuda()
		fake_datav = autograd.Variable(fake_data)
		fake = autograd.Variable(netG(fake_datav).data)
		G = netD(fake)
		G.requires_grad = True
		G = G.mean()
		# G.backward(mone)
		G.backward()
		G_cost = -G
		optimizerG.step()

		print("G_cost" , G_cost)
		print("D_cost", D_cost)
if __name__ == "__main__":
	main()