import torch
from torch import nn
import numpy as np
import sys
sys.path.append("./ProgressiveGAN/EEG-GAN/")
import eeggan
from eeggan.examples.conv_lin.model import Generator, Discriminator

def load_model(d_filename, g_filename, n_z=200):
	d = Discriminator(1)
	g = Generator(1, n_z)
	d.load_state_dict(torch.load(d_filename))
	g.load_state_dict(torch.load(g_filename))
	return d, g

def generate_noise(batch_size, task_index=0, n_z=200):
	rng = np.random.RandomState(task_index)
	z_vars = rng.normal(0, 1, size=(batch_size, n_z)).astype(np.float32)
	z_vars = torch.autograd.Variable(torch.from_numpy(z_vars), requires_grad=False)
	return z_vars

def save_reading(readings, file):
	# readings - (num samples, num signals)
	np.save(file, readings)

def generate_readings(suffix, num_examples, filepath="./ProgressiveGAN/EEG-GAN/eeggan/examples/conv_lin/", prefix_d="discriminator", prefix_g="generator", block=5):
	d_filename = filepath + prefix_d + suffix + ".pt"
	g_filename = filepath + prefix_g + suffix + ".pt"
	_, g = load_model(d_filename, g_filename)
	g.model.cur_block = block
	z = generate_noise(num_examples)
	samples = np.squeeze(g(z).detach().numpy())
	return samples

if __name__ == "__main__":
	# d_filename = "./ProgressiveGAN/EEG-GAN/eeggan/examples/conv_lin/discriminator-s.pt"
	# g_filename = "./ProgressiveGAN/EEG-GAN/eeggan/examples/conv_lin/generator-s.pt"
	# _, g = load_model(d_filename, g_filename)
	# x = generate_noise(64)
	# readings = np.squeeze(g(x).detach().numpy())
	# print("readings:", readings.shape)
	samples = generate_readings("", 20)
	print("samples shape", samples.shape)

