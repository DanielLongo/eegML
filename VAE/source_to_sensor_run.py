# adapted from https://github.com/coolvision/vae_conv/blob/master/vae_conv_mnist.py
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from source_to_sensor_model import ForwardLearned
from forward_model_dataloader import ForwardModelDataset

num_epochs = 10000
dataset = ForwardModelDataset(64*4, batch_size=64, length=768)
use_cuda = torch.cuda.is_available()

model = ForwardLearned()
if use_cuda:
	model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    # print(recon_x.size(), x.size())
    BCE = F.binary_cross_entropy(recon_x.view(-1, 44 * 768), x.view(-1, 44 * 768), size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return BCE + KLD
    return BCE + 3 * KLD

def train(num_epochs):
    model.train()
    train_loss = 0
    iters = 0
    print_iter = 10
    m = nn.Sigmoid()
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
            recon_batch = m(recon_batch)
            s_t = m(s_t)
            loss = loss_function(recon_batch, s_t, mu, logvar)
            loss.backward()
            cur_epoch_loss += loss.item()
            cur_epoch_iters += 1
            optimizer.step()
            # if iters % print_iter == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, iters , len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader),
            #         loss.data[0] / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              i, cur_epoch_loss / cur_epoch_iters))

        # if epoch % 10 == 0:
        torch.save(model.state_dict(),  "VAE-" + "first" + ".pt")

train(num_epochs)