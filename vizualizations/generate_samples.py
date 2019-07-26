import torch
from torch import nn
import numpy as np
import sys

sys.path.append("../ProgressiveGAN/EEG-GAN/")
sys.path.append("../VAE/")
import eeggan
from source_to_sensor_model import ForwardLearned
from eeggan.examples.conv_lin.model import Generator, Discriminator
from sensor_to_sensor_model import SensorToSensor


def load_model(d_filename, g_filename, n_z=200, map_location='gpu', num_channels=1):
    print("map location", map_location)
    d = Discriminator(num_channels)
    g = Generator(num_channels, n_z)
    d.load_state_dict(torch.load(d_filename, map_location=map_location))
    g.load_state_dict(torch.load(g_filename, map_location=map_location))
    return d, g


def generate_noise(batch_size, task_index=0, n_z=200):
    # rng = np.random.RandomState(task_index)
    rng = np.random.RandomState(None)
    z_vars = rng.normal(0, 1, size=(batch_size, n_z)).astype(np.float32)
    # print("z_vars_sum", np.sum(z_vars))
    z_vars = torch.autograd.Variable(torch.from_numpy(z_vars), requires_grad=False)
    return z_vars


def generate_z_x_t(batch_size, num_dipoles=7498, time=768):
    rng = np.random.RandomState(None)
    z_vars = rng.normal(0, 1, size=(batch_size, num_dipoles, time)).astype(np.float32)
    z_vars = torch.autograd.Variable(torch.from_numpy(z_vars), requires_grad=False)
    return z_vars


def save_reading(readings, file):
    # readings - (num samples, num signals)
    np.save(file, readings)


def generate_readings(suffix, num_examples, filepath="../ProgressiveGAN/EEG-GAN/eeggan/examples/conv_lin/",
                      prefix_d="discriminator", prefix_g="generator", block=5, num_channels=1, use_cuda=True):
    d_filename = filepath + prefix_d + suffix + ".pt"
    g_filename = filepath + prefix_g + suffix + ".pt"
    if torch.cuda.is_available() and use_cuda:
        _, g = load_model(d_filename, g_filename, num_channels=num_channels)
    else:
        _, g = load_model(d_filename, g_filename, map_location='cpu', num_channels=num_channels)
    print("d_filename", d_filename)
    g.model.cur_block = block
    z = generate_noise(num_examples)
    samples = np.squeeze(g(z).detach().numpy())
    return samples


def generate_readings_from_x_t(num_examples, filename, filepath="../VAE/", num_dipoles=7498, t=768, use_gpu=True):
    g = ForwardLearned()
    z = generate_z_x_t(num_examples)
    if torch.cuda.is_available() and use_gpu:
        print("loading on a GPU")
        g.load_state_dict(torch.load(filepath + filename + ".pt", map_location='gpu'))
    else:
        print("loading on a CPU")
        g.load_state_dict(torch.load(filepath + filename + ".pt", map_location=lambda storage, loc: storage))
    print("loading samples")
    print("type of z", type(z))
    g.cuda()
    z = z.cuda()
    samples, _, _ = g(z)
    return np.squeeze(samples.cpu().detach().numpy())

def generate_readings_from_s_t(num_examples, filename, filepath="../VAE/", use_gpu=True):
    g = SensorToSensor(num_channels=1)
    if torch.cuda.is_available() and use_gpu:
        g.load_state_dict(torch.load(filepath + filename + ".pt", map_location='gpu'))
    else:
        g.load_state_dict(torch.load(filepath + filename + ".pt", map_location='cpu'))

    z = generate_noise(num_examples)
    samples = np.squeeze(g.generate(z).detach().numpy())
    return samples


if __name__ == "__main__":
    # d_filename = "./ProgressiveGAN/EEG-GAN/eeggan/examples/conv_lin/discriminator-s.pt"
    # g_filename = "./ProgressiveGAN/EEG-GAN/eeggan/examples/conv_lin/generator-s.pt"
    # _, g = load_model(d_filename, g_filename)
    # x = generate_noise(64)
    # readings = np.squeeze(g(x).detach().numpy())
    # print("readings:", readings.shape)
    samples = generate_readings("-s", 20)
    print("samples shape", samples.shape)
