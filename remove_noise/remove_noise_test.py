import torch
import numpy as np
from torch import nn
import sys
from remove_noise_conv import ConvRemoveNoise
from estimated_loader import EstimatedEEGs
from torch.autograd import Variable

sys.path.append("./generators/")
sys.path.append("./discriminators/")

cuda = True
num_epochs = 100
batch_size = 64
num_batches = 100
print_iter = 1
iters = 0

cleaner = ConvRemoveNoise(44, 50)
estimated_eegs = EstimatedEEGs(num_channels=44, length=1004, batch_size=batch_size)
optimizer_C = torch.optim.Adam(cleaner.parameters(), lr=.01)
critereon = torch.nn.L1Loss()

if cuda:
	cleaner.cuda()

for epoch in range(num_epochs):
	for i in range(num_batches):
		iters += 1
		estimated = estimated_eegs[i] * 1e5 * 4
		# estimated = torch.ones(batch_size, 1004, 44)
		#random
		noise = torch.randn(batch_size, 1004, 44)#  * 1e-5 * .2
		#sin wave
		# noise = torch.from_numpy(np.sin(np.tile(np.tile(np.arange(1004), [44,1]).T, [64,1,1]))).float()  * 2#* 1e-5 * .2

		if cuda:
			estimated = estimated.cuda()
			noise = noise.cuda()

		noisy_eegs = estimated + noise
		
		optimizer_C.zero_grad()
		# estimated = estimated.view(batch_size, 1, 1004, 44)
		print("estimated", np.sum(np.abs(estimated.cpu().detach().numpy())))
		print("noise", np.sum(np.abs(noise.cpu().detach().numpy())))
		print("Before", torch.sum(noisy_eegs - estimated))
		cleaned_noisy = Variable(cleaner(noisy_eegs), requires_grad=True)
		cleaned_clean = Variable(cleaner(estimated), requires_grad=True)
		print("After", torch.sum(cleaned_noisy - cleaned_clean))
		print("cleaned clean and estimated", torch.sum(cleaned_clean - estimated))
		print("estimated and estimated", torch.sum(estimated - estimated))
		noisy_loss = torch.dist(cleaned_noisy, estimated)
		clean_loss = torch.dist(cleaned_clean, estimated)
		# clean_loss = critereon(cleaned_clean, estimated)
		# noisy_loss = critereon(cleaned_noisy, estimated)

		cleaner_loss = noisy_loss + clean_loss
		# cleaner_loss = clean_loss
		cleaner_loss.backward()
		optimizer_C.step()

		if iters % print_iter == 0:
			print("cleaner loss @ iter", iters, "total loss", cleaner_loss.cpu(), "loss noisy", noisy_loss.cpu(), "clean noisy", clean_loss.cpu())


