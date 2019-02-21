import torch
from torch import nn
import sys
from remove_noise_conv import ConvRemoveNoise
from estimated_loader import EstimatedEEGs
from torch.autograd import Variable

sys.path.append("./generators/")
sys.path.append("./discriminators/")

cuda = False
num_epochs = 100
batch_size = 64
num_batches = 100
print_iter = 1000
iters = 0

cleaner = ConvRemoveNoise(44, 50)
estimated_eegs = EstimatedEEGs(num_channels=44, length=1004, batch_size=batch_size)
optimizer_C = torch.optim.Adam(cleaner.parameters())

if cuda:
	cleaner.cuda()
	optimizer_C = optimizer_C.cuda()

for epoch in range(num_epochs):
	for i in range(num_batches):
		iters += 1
		estimated = estimated_eegs[i]
		noise = torch.randn(batch_size, 1004, 44) * 3

		if cuda:
			estimated = estimated.cuda()
			noise = noise.cuda()

		noisy_eegs = estimated + noise
		
		optimizer_C.zero_grad()

		cleaned_noisy = Variable(cleaner(noisy_eegs), requires_grad=True)
		cleaned_clean = Variable(cleaner(estimated), requires_grad=True)
		noisy_loss = torch.dist(cleaned_noisy, estimated)
		clean_loss = torch.dist(cleaned_clean, estimated)

		cleaner_loss = noisy_loss + clean_loss
		cleaner_loss.backward()
		optimizer_C.step()

		if iters % print_iter == 0:
			print("cleaner loss @ iter", iters, "total loss", cleaner_loss.cpu(), "loss noisy", noisy_loss.cpu(), "clean noisy", clean_loss.cpu())


