import torch
import time
from torch import nn
import torchvision.datasets
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.autograd as autograd
import torch.nn.functional as F

plt.rcParams['image.cmap'] = 'gray'

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
def generate_nosie(batch_size, dim=100):
	noise = torch.randn(batch_size, dim, 1, 1)
	return noise

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 32, [5,5], stride=[1,1]),
			nn.LeakyReLU(negative_slope=.01),
			nn.MaxPool2d([2,2], stride=[2,2]))
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, [5,5], stride=[1,1]),
			nn.LeakyReLU(negative_slope=.01),
			nn.MaxPool2d([2,2], stride=[2,2]))
		self.fc1 = nn.Sequential(
			nn.Linear((64*5*5), (64*5*5)),
			nn.LeakyReLU(negative_slope=.01))
		self.fc2 = nn.Sequential(
			nn.Linear((64*5*5), 1),
			nn.Sigmoid())


	def forward(self, x, matching=False):
		out = self.conv1(x)
		out = self.conv2(out)
		out = out.view(out.shape[0], -1)
		out = self.fc1(out)
		final = self.fc2(out)
		if matching:
			return out, final
		return final

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.deconv1 = nn.Sequential(
			nn.ConvTranspose2d(100, 128, [2,2], stride=[1,1]),
			nn.BatchNorm2d(128),
			nn.ReLU())
		self.deconv2 = nn.Sequential(
			nn.ConvTranspose2d(128, 256, [3,3], stride=[1,1]),
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.deconv3 = nn.Sequential(
			nn.ConvTranspose2d(256, 256, [4,4], stride=[2,2], padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.deconv4 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, [4,4], stride=[2,2], padding=1),
			nn.BatchNorm2d(128))
		self.deconv5 = nn.Sequential(
			nn.ConvTranspose2d(128, 1, [4,4], stride=[2,2], padding=1),
			nn.Tanh())

	def forward(self, x):
		# print("x", x.shape)
		out = self.deconv1(x)
		out = self.deconv2(out)
		out = self.deconv3(out)
		out = self.deconv4(out)
		out = self.deconv5(out)
		return out


	def weight_init(m, mean, std):
		if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
			m.weight.data.normal_(mean, std)
			m.bias.data.zero_()

def create_optimizer(model, lr=.01, betas=None):
	if betas == None:
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
	return optimizer

def save_images_func(generator, epoch, i, filename_prefix):
	fig = plt.figure(figsize=(10, 10))
	gs = gridspec.GridSpec(10, 10)
	gs.update(wspace=.05, hspace=.05)
	z = generate_nosie(100)
	images_fake = generator(z)
	images_fake = images_fake.data.data.cpu().numpy()
	for img_num, sample in enumerate(images_fake):
		ax = plt.subplot(gs[img_num])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(32, 32), cmap='Greys_r')

	filename = filename_prefix + str(epoch) + "-" + str(i) 
	plt.savefig("./generated_images/" + filename, bbox_inches="tight" )
	plt.close(fig)

def train_gan(discriminator, generator, image_loader, num_epochs, batch_size, g_lr, d_lr, dtype, filename_prefix="DCGAN-", save_images=True):
	iters = 0
	optimizerD = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
	optimizerG = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
	criterionD = nn.CrossEntropyLoss().cuda() # binary cross-entropy
	criterionG = nn.MSELoss().cuda()
	iters = 0
	for epoch in range(num_epochs):
		for x, _ in image_loader:
			if (x.shape[0] != batch_size):
				continue
			iters += 1
			############################
			# (1) Update D network
			###########################
			for p in discriminator.parameters():  # reset requires_grad
				p.requires_grad = True  # they are set to False below in discriminator update:

			# for p in discriminator.parameters():
			# 	p.requires_grad = False # to avoid computation
			
			discriminator.zero_grad()
			inputv = autograd.Variable(x)

			inputv = inputv.type(dtype)

			# print("inputv", inputv.shape)

			unl_output = discriminator(inputv)
			loss_unl_real = -torch.mean(LSE(unl_output),0) +  torch.mean(F.softplus(LSE(unl_output),1),0)
			real_loss = unl_output

			# train with fake
			# noise = discriminator.generate_noise(batch_size, LENGTH, NUM_NODES)
			noise = generate_nosie(x.shape[0])

			noise = noise.type(dtype)

			noise_v = autograd.Variable(noise)

			fake = autograd.Variable(generator(noise_v).data)
			unl_output = discriminator(fake.detach()) #fake images are separated from the graph #results will never gradient(be updated), so G will not be updated
			loss_unl_fake = torch.mean(F.softplus(LSE(unl_output),1),0)
			fake_loss = unl_output
			# loss_D = loss_unl_real + loss_unl_fake
			# loss_D = real_loss - fake_loss
			loss_D = torch.mean(real_loss) - torch.mean(fake_loss)
			loss_D.backward()# because detach(), backward() will not influence discriminator
			optimizerD.step()

		############################
		# (2) Update G network
		###########################
			for p in discriminator.parameters():
				p.requires_grad = False  # to avoid computation

			# for p in discriminator.parameters():
			# 	p.requires_grad = True

			discriminator.zero_grad()

			# noise = discriminator.generate_noise(batch_size, LENGTH, NUM_NODES)
			noise = generate_nosie(x.shape[0])

			noise = noise.type(dtype)

			noise_v = autograd.Variable(noise)

			fake = autograd.Variable(generator(noise_v).data)
			
			inputv = autograd.Variable(x).type(dtype)
			feature_real,_ = discriminator(inputv, matching=True)
			feature_fake,output = discriminator(fake, matching=True)
			feature_real = torch.mean(feature_real,0)
			feature_fake = torch.mean(feature_fake,0)
			loss_G = criterionG(feature_fake, feature_real.detach())
			optimizerG.step()
			# if (iters % 1000 == 0):
			# 	save_EEG(fake.cpu().detach().numpy(), NUM_NODES, 200, "./generated_eegs/generated-iter"+ str(iters) + "-fake-rG-long")
			# 	print("Epoch", iteration)
			# 	print("G_cost" , G_cost)
			# 	print("D_cost", D_cost)
		# save_EEG(fake.cpu().detach().numpy(), NUM_NODES, 200, "./generated_eegs/generated-" + str(iteration) + "-fake-cG-matching-minB")
		# save_EEG(real.cpu().detach().numpy(), NUM_NODES, 200, "./generated_eegs/generated-" + str(iteration-1) + "-real-rG-long-norm")
		save_images_func(generator, epoch, iters, filename_prefix)
		print("Epoch", epoch)
		print("G_cost" , loss_G)
		print("D_cost", loss_D)
if __name__ == "__main__":
	filename = "dcgan"
	d_filename = "D_mnist"
	g_filename = "G_mnist"
	batch_size = 128
	img_size = 32
	num_epochs = 10
	lr = .0002
	discriminator = Discriminator()
	generator = Generator()
	if torch.cuda.is_available():
		print("Running On GPU :)")
		torch.set_default_tensor_type("torch.cuda.FloatTensor")
		torch.backends.cudnn.benchmark = True
		dtype = torch.cuda.FloatTensor
		use_cuda = True
		discriminator = discriminator.cuda()
		generator = generator.cuda()
	else:
		print("Running On CPU :(")
		print("This may take a while")
		use_cuda = False
		dtype = torch.FloatTensor

	transform = transforms.Compose([
		transforms.Resize(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
	mnist_test = torchvision.datasets.MNIST('./MNIST_data', train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,  shuffle=True)

	train_gan(discriminator, generator, train_loader, num_epochs, batch_size, lr, lr, dtype, save_images=True)
