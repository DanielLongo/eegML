# adapted from https://github.com/coolvision/vae_conv/blob/master/vae_conv_mnist.py
import torch
import numpy as np
import sys
sys.path.append("../data_loaders/")
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from sensor_to_sensor_model import SensorToSensor
from load_eegs_one_c_improved import EEGDataset

num_epochs = 100000
dataset = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=64*7, num_channels=44, length=768, csv_file=None)
# dataset = ForwardModelDataset(4, batch_size=2, length=768)
use_cuda = torch.cuda.is_available()

model = SensorToSensor(num_channels=1)
if use_cuda:
	model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=1e-2)
optimizer = optim.Adam(model.parameters())
# optimizer = optim.Adam(model.parameters(), lr=1e-6)

def loss_function(recon_x, x, mu, logvar, print_=False):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False) #, reduction="mean")# * 1e-6
    BCE = F.mse_loss(recon_x, x, size_average=False)
    # BCE *= 1e-5
# see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD *= 1e-1
    KLD *= 3
    # KLD = 0
    # if print_:
        # print("KLD", KLD, "BCE ", BCE)
# return BCE + KLD
    return (BCE + KLD) * .015

def autoencoder_loss(recon_x, x):
    BCE = F.mse_loss(recon_x, x, size_average=False)
    # print("BCE", BCE)
    return BCE

def normalize(batch):
    batch = batch - batch.mean()
    batch = batch / batch.std()
    batch = batch / np.abs(batch).max()
    return batch

def train(num_epochs):
    model.train()
    costs = []
    train_loss = 0
    iters = 0
    print_iter = 1
    print_ = False
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    m = nn.Sigmoid()
    for i in range(num_epochs):
        cur_epoch_loss = 0
        cur_epoch_iters = 0
        # lr *= .99
        for j in range(len(dataset)):
            iters += 1
            if iters % print_iter == 0:
                print_ = True
            else:
                print_ = False
            s_t = dataset[j]
            if use_cuda:
                s_t = s_t.cuda()
            optimizer.zero_grad()
            # recon_batch, mu, logvar = model(s_t)
            # recon_batch = torch.transpose(recon_batch.squeeze(), 2, 1)
            recon_batch, mu, logvar = model(s_t)
            recon_batch = recon_batch.squeeze()
            s_t = s_t.squeeze()
            if i % 50 == 0 and j == 0:
                np.save("samples_sensor_to_sensor/test-5-recon", recon_batch.detach().cpu().numpy())
                np.save("samples_sensor_to_sensor/test-5-original", s_t.detach().cpu().numpy())
                np.save("StS-test-5", np.asarray(costs))
                torch.save(model.state_dict(),  "VAE-" + "StS-test-5-BIGgrad" + ".pt")
            s_t = m(s_t)
            recon_batch = m(recon_batch)
            loss = loss_function(recon_batch, s_t, mu, logvar, print_=print_)
            loss.backward()
            cur_epoch_loss += float(loss.cpu().item())
            cur_epoch_iters += 1
            optimizer.step()
            # if iters % print_iter == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, iters , len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader),
            #         loss.data[0] / len(data)))
        costs += [cur_epoch_loss / cur_epoch_iters]
        if print_:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
              i, cur_epoch_loss / cur_epoch_iters))

train(num_epochs)
print("finished StS-test-5-BIGgrad")
