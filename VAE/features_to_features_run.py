# adapted from https://github.com/coolvision/vae_conv/blob/master/vae_conv_mnist.py
import torch
import numpy as np
import sys
import time
from torch.utils.data import DataLoader
sys.path.append("../tsy935/RubinLab_neurotranslate_eeg-master/eeg/data/")
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from features_to_features_model import FeaturesToFeatures
from data_loader import SeizureDataset

num_epochs = 100000

batch_size = 7 # 7 b/c 7 * 9 = 63
train_dataset = SeizureDataset()
print("len train", len(train_dataset))
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
use_cuda = torch.cuda.is_available()


model = FeaturesToFeatures(pretrained_encoder=False)
test_num = 6

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())


def recons_loss(recon_x, x):
    # d = 1 if len(recon_x.shape) == 2 else (1, 2, 3)

    return F.mse_loss(recon_x, x,  reduction='sum')
    # return F.mse_loss(recon_x, x, size_average=False)

def vae_gaussian_kl_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    return torch.sum(KLD)


def loss_function(recon_x, x, mu, logvar, print_=False):
    likelihood = recons_loss(recon_x, x)
    kl_loss = vae_gaussian_kl_loss(mu, logvar)
    vae_loss = (likelihood + kl_loss * 5)
    return torch.sum(vae_loss)


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
    print_iter = 20
    print_ = True
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(num_epochs):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        cur_epoch_loss = 0
        cur_epoch_iters = 0
        lr *= .995
        for features, y, _, in train_loader:
            iters += 1

            if features.shape[0] != batch_size or features.shape[1] != 9:
                continue

            if iters % print_iter == 0:
                print_ = True
            else:
                print_ = False
            if use_cuda:
                features = features.cuda()

            features = features.view(batch_size * 9, 3, 224, 224)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(features)
            recon_batch = recon_batch.squeeze()
            features = features.squeeze()
            if i % 50 == 0:
                np.save("samples_features_to_features/test-" + str(test_num) + "-recon", recon_batch.detach().cpu().numpy())
                np.save("samples_features_to_features/test-" + str(test_num) + "-original", features.detach().cpu().numpy())
                np.save("StS-test-5", np.asarray(costs))
                torch.save(model.state_dict(), "VAE-" + "FtF-test-" + str(test_num) + "-csv" + ".pt")
            # s_t = m(s_t)
            # recon_batch = m(recon_batch)
            loss = loss_function(recon_batch, features, mu, logvar, print_=print_)
            loss.backward()
            cur_epoch_loss += float(loss.cpu().item())
            cur_epoch_iters += 1
            optimizer.step()
            if iters % print_iter == 0 and print_:
                print(" Train Epoch: {} [loss = {}, iters = {}]". format(
                    i, loss.item(), iters))
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             i, iters , len(train_loader.dataset),
        #             100. * batch_idx / len(train_loader),
        #             loss.data[0] / len(data)))
        # costs += [cur_epoch_loss / cur_epoch_iters]
        if print_:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                i, cur_epoch_loss / cur_epoch_iters))


train(num_epochs)
print("finished" + str(test_num))