import random
import torch
import sys

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("./generators/")
sys.path.append("./discriminators/")
from rDiscriminator import RecurrentDiscriminator
from rD_min_batch import rDMinBatch
from forward_model_enabled_G import ForwardModelEnabledG
from conditional_generator import ConditionalGenerator
from c_forward_model import cGForwardModel
from rG import RecurrentGenerator
from load_EEGs import EEGDataset
from utils import save_EEG

USE_CUDA = True
PARALLEL = True
ITERS = 30
CRITIC_ITERS = 2 #3
BATCH_SIZE = 32
LAMBDA = 10 # Gradient penalty lambda hyperparameter
NUM_NODES = 42
# LENGTH = 1000
LENGTH = 1000

real_eegs = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=5000, num_channels=NUM_NODES, batch_size=BATCH_SIZE, length=1000, delay=100000)#000)


one = torch.ones([])
mone = one * -1

## Pick Of Discriminators 
# netD = RecurrentDiscriminator(num_nodes=NUM_NODES, d=64)
netD = rDMinBatch(num_nodes=NUM_NODES, d=64)

## Pick Of Generators
# netG = RecurrentGenerator(num_nodes=NUM_NODES, d=50) 
# netGG = netG
# netG = ForwardModelEnabledG(44, 50)
netG = ConditionalGenerator(num_nodes=42, d=64, y_input_size=20)
# netG = cGForwardModel(num_nodes=44, d=64, y_input_size=20)

noise_gen_G = netG.generate_noise
if PARALLEL:
	netD = nn.DataParallel(netD)
	netG = nn.DataParallel(netG)

optimizerD = optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=1e-5, betas=(0.5, 0.99))

criterionD = nn.CrossEntropyLoss() # binary cross-entropy
criterionG = nn.MSELoss()
if USE_CUDA:
	one = one.cuda()
	mone = mone.cuda()
	netD.cuda()
	netG.cuda()
	criterionG.cuda()
	criterionD.cuda()


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

def LSE(before_softmax_output):
	# exp = torch.exp(before_softmax_output)
	# sum_exp = torch.sum(exp,1) #right
	# log_sum_exp = torch.log(sum_exp)
	# return log_sum_exp
	vec = before_softmax_output
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	output = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast),1))
	return output
	
def to_scalar(var):
	# returns a python float
	return var.view(-1).data.tolist()[0]

def argmax(vec):
	# return the argmax as a python int
	_, idx = torch.max(vec, 1)
	return to_scalar(idx)

def main():
	iters = 0
	for iteration in range(ITERS):
		real_eegs.shuffle()
		for i in range(len(real_eegs)):
			if (real_eegs[i].shape[0] != BATCH_SIZE):
				continue
			iters += 1
			############################
			# (1) Update D network
			###########################
			for p in netD.parameters():  # reset requires_grad
				p.requires_grad = True  # they are set to False below in netG update:

			# for p in netG.parameters():
			# 	p.requires_grad = False # to avoid computation
			
			optimizerD.zero_grad()
			inputv = autograd.Variable(real_eegs[i])

			if USE_CUDA:
				inputv = inputv.cuda()

			unl_output = netD(inputv)
			loss_unl_real = -torch.mean(LSE(unl_output),0) +  torch.mean(F.softplus(LSE(unl_output),1),0)

			# train with fake
			# noise = netG.generate_noise(BATCH_SIZE, LENGTH, NUM_NODES)
			noise = noise_gen_G(BATCH_SIZE, LENGTH, NUM_NODES)

			if USE_CUDA:
				noise = [x.cuda() for x in noise]

			noise_v = [autograd.Variable(x) for x in noise]

			if (len(noise) == 1): #not conditional
				fake = autograd.Variable(netG(noise_v[0]).data)
			if (len(noise) == 2): #conditional with 1 Y
				fake = autograd.Variable(netG(noise_v[0], noise_v[1]).data)
			if (len(noise) == 3): #conditional with 2 Ys
				fake = autograd.Variable(netG(noise_v[0], noise_v[1], noise_v[2]).data)

			unl_output = netD(fake.detach()) #fake images are separated from the graph #results will never gradient(be updated), so G will not be updated
			loss_unl_fake = torch.mean(F.softplus(LSE(unl_output),1),0)
			loss_D = loss_unl_real + loss_unl_fake
			loss_D.backward()# because detach(), backward() will not influence netG
			optimizerD.step()

		############################
		# (2) Update G network
		###########################
			for p in netD.parameters():
				p.requires_grad = False  # to avoid computation

			# for p in netG.parameters():
			# 	p.requires_grad = True

			optimizerG.zero_grad()

			# noise = netG.generate_noise(BATCH_SIZE, LENGTH, NUM_NODES)
			noise = noise_gen_G(BATCH_SIZE, LENGTH, NUM_NODES)	

			if USE_CUDA:
				noise = [x.cuda() for x in noise]

			noise_v = [autograd.Variable(x) for x in noise]

			if (len(noise) == 1): #not conditional
				fake = autograd.Variable(netG(noise_v[0]).data)
			if (len(noise) == 2): #conditional with 1 Y
				fake = autograd.Variable(netG(noise_v[0], noise_v[1]).data)
			if (len(noise) == 3): #conditional with 2 Ys
				fake = autograd.Variable(netG(noise_v[0], noise_v[1], noise_v[2]).data)
			
			inputv = autograd.Variable(real_eegs[i])
			feature_real,_ = netD(inputv, matching=True)
			feature_fake,output = netD(fake, matching=True)
			feature_real = torch.mean(feature_real,0)
			feature_fake = torch.mean(feature_fake,0)
			loss_G = criterionG(feature_fake, feature_real.detach())
			optimizerG.step()
			# if (iters % 1000 == 0):
			# 	save_EEG(fake.cpu().detach().numpy(), NUM_NODES, 200, "./generated_eegs/generated-iter"+ str(iters) + "-fake-rG-long")
			# 	print("Epoch", iteration)
			# 	print("G_cost" , G_cost)
			# 	print("D_cost", D_cost)
		save_EEG(fake.cpu().detach().numpy(), NUM_NODES, 200, "./generated_eegs/generated-" + str(iteration) + "-fake-cG-matching-minB")
		# save_EEG(real.cpu().detach().numpy(), NUM_NODES, 200, "./generated_eegs/generated-" + str(iteration-1) + "-real-rG-long-norm")
		print("Epoch", iteration)
		print("G_cost" , loss_G)
		print("D_cost", loss_D)
if __name__ == "__main__":
	main()
