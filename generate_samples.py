import torch
from torch import nn
from eeggan.examples.conv_lin.model import Generator, Discriminator
import sys

def load_file(d_filenamem g_filename, n_z=200):
	d = Discriminator(1, n_z)
	g = Generator(1)
	d.load_state_dict(torch.load(d_filenamem))
	g.load_state_dict(torch.load(g_filename))
	return d, g

def generate_noise(batch_size, task_index=0):
	rng = np.random.RandomState(task_index)
	z_vars = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)
	z_vars = Variable(torch.from_numpy(z_vars), requires_grad=False)
	return z_vars

if __name__ == "__main__":
	x = generate_noise(100)
	print(x.shape)