import torch
from rGAN import RecurrentGenerator, RecurrentDiscriminator

ITERS = 10
CRITIC_ITERS = 3

noisy_eegs = torch.randn(16, 4, 100, 32) #None, batch_size, time, num_nodes
clean_eegs = torch.randn(16, 4, 100, 32)

for iteration in xrange(ITERS):
	for i in range(noisy_eegs.shape[0])
		############################
		# (1) Update D network
		###########################
		for p in netD.parameters():  # reset requires_grad
			p.requires_grad = True  # they are set to False below in netG update

		for iter_d in xrange(CRITIC_ITERS):
			real_data = noisy_eegs[i*iter_d]
			if use_cuda:
				real_data = real_data.cuda(gpu)
			real_data_v = autograd.Variable(real_data)

			netD.zero_grad()

			# train with real
			D_real = netD(real_data_v)
			D_real = D_real.mean()
			# print D_real
			D_real.backward(mone)

			# train with fake
			fake_data = clean_eegs[i*iter_d]
			if use_cuda:
				fake_data = noise.cuda(gpu)
			fake_datav = autograd.Variable(fake_data, volatile=True)  # totally freeze netG
			fake = autograd.Variable(netG(fake_datav).data)
			inputv = fake
			D_fake = netD(inputv)
			D_fake = D_fake.mean()
			D_fake.backward(one)

			# train with gradient penalty
			gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
			gradient_penalty.backward()

			D_cost = D_fake - D_real + gradient_penalty
			Wasserstein_D = D_real - D_fake
			optimizerD.step()

		############################
		# (2) Update G network
		###########################
		for p in netD.parameters():
			p.requires_grad = False  # to avoid computation
		netG.zero_grad()

		fake_data = clean_eegs[i]
		if use_cuda:
			fake_data = noise.cuda(gpu)
		fake_datav = autograd.Variable(fake_data)
		fake = autograd.Variable(netG(fake_datav).data)
		G = netD(fake)
		G = G.mean()
		G.backward(mone)
		G_cost = -G
		optimizerG.step()
