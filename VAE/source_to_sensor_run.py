# adapted from https://github.com/coolvision/vae_conv/blob/master/vae_conv_mnist.py
import torch
import numpy as np
import sys
sys.path.append("../data_loaders/")
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from source_to_sensor_model import ForwardLearned
from forward_model_dataloader import ForwardModelDataset

num_epochs = 100000
dataset = ForwardModelDataset(64*3, batch_size=64, length=768)
# dataset = ForwardModelDataset(4, batch_size=2, length=768)
use_cuda = torch.cuda.is_available()

model = ForwardLearned()
if use_cuda:
	model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-6)

def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x.view(-1, 44 * 768), x.view(-1, 44 * 768), size_average=False) * 1e-6
    print(recon_x.shape, x.shape)
    MSE = F.mse_loss(recon_x, x)
    MSE *= 1e-7
# see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD *= 3e-1
    KLD = 0
    print("KLD", KLD, "MSE Adjusted", MSE)
# return BCE + KLD
    return MSE + KLD

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
    print_iter = 10
    # m = nn.Sigmoid()
    for i in range(num_epochs):
        cur_epoch_loss = 0
        cur_epoch_iters = 0
        for j in range(len(dataset)):
            iters += 1
            x_t = dataset.getSources(j)
            s_t = (dataset.getEEGs(j))
            x_t = Variable(x_t)
            if use_cuda:
                x_t = x_t.cuda()
                s_t = s_t.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_t)
            recon_batch = torch.transpose(recon_batch.squeeze(), 2, 1)
            recon_batch = recon_batch
            # recon_batch = m(recon_batch)
            # s_t = m(s_t)
            loss = loss_function(recon_batch, s_t, mu, logvar)
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
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              i, cur_epoch_loss / cur_epoch_iters))
        if i % 10 == 0:
            np.save("samples/sample-recon_no_KLD", recon_batch.detach().cpu().numpy())
            np.save("samples/sample-s_t_no_KLD", s_t.detach().cpu().numpy())
            np.save("test-7-no_KLD", np.asarray(costs))
            torch.save(model.state_dict(),  "VAE-" + "test-7-no_KLD" + ".pt")

train(num_epochs)
print("finished test-7-no_KLD")
