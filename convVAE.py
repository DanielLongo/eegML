import torch
import numpy as np
from torch import nn
import sys
from estimated_loader import EstimatedEEGs
from torch.autograd import Variable
from utils import save_EEG

sys.path.append("./generators/")
sys.path.append("./discriminators/")

from convG_eeg import ConvGenerator
from load_EEGs_improved import EEGDataset

cuda = True
num_epochs = 2000
batch_size = 64
num_batches = 100
print_iter = 10
latent_dim = 100
img_shape = (1, 32, 32)


class ConvEncoder(nn.Module):
    # input shape (seq_len, batch, input_size)
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.channels_out = 1
        self.channels_h = 1

        self.conv_blocks_encode = nn.Sequential(
            nn.Conv2d(self.channels_out, 4, [3, 4], stride=[2, 3], padding=0),
            nn.BatchNorm2d(4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 16, [6, 4], stride=[2, 1], padding=0),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, self.channels_h, [3, 3], stride=[2, 2], padding=0),
            nn.BatchNorm2d(self.channels_h, 0.8),
            nn.Tanh()
        )

    def forward(self, x):
        if (len(x.shape) == 3):
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        encoded = self.conv_blocks_encode(x)
        return encoded


# estimated_eegs = EstimatedEEGs(num_channels=44, length=1004, batch_size=batch_size)
# data_file = "/mnt/data1/eegdbs/all_reports_impress_blanked-2019-02-23.csv"
data_file = None
real_eegs = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", csv_file=data_file, num_examples=64 * 6, num_channels=44,
                       length=1004, delay=10000)
print("loaded")
# critereon = torch.nn.L1Loss()
critereon = torch.nn.MSELoss()

encoder = ConvEncoder()
decoder = ConvGenerator(img_shape, latent_dim)
net = nn.Sequential(
    ConvEncoder(),
    ConvGenerator(img_shape, latent_dim)
)

if cuda:
    net.cuda()
optim = torch.optim.Adam(net.parameters(), lr=1 * 1e-4)

def normalize(batch):
    batch = batch - batch.mean()
    batch = batch / batch.std()
    batch = batch / np.abs(batch).max()
    return batch

def main():
    iters = 0
    costs = []
    for epoch in range(num_epochs):
        costs_per_epoch = []
        real_eegs.shuffle()
        for i, eegs in enumerate(real_eegs):
            if (eegs.shape[0] != batch_size):
                continue
            eegs = normalize(eegs)
            iters += 1
            optim.zero_grad()
            # eeg = estimated_eegs[i]
            eeg = Variable(eegs)
            if cuda:
                eeg = eeg.cuda()
            # eeg *= 1e5 *4
            x_prime = net(eeg)
            cost = critereon(x_prime, eeg) * 1e3
            cost.backward()
            optim.step()
            costs_per_epoch += [cost.item()]
            if iters % print_iter == 0:
                avg_cost_epoch = sum(costs_per_epoch) / len(costs_per_epoch)
                print("[Iter: " + str(iters) + "] [Epoch: " + str(epoch) + "] [Avg cost in epoch %f ] [Loss: %f]" % (
                avg_cost_epoch, cost.item()))
                save_EEG(eeg.cpu().detach().view(batch_size, 1004, 44).numpy(), 44, 200,
                         "./reonconstructed_eegs/E-orginal-" + str(epoch))
                save_EEG(x_prime.cpu().detach().view(batch_size, 1004, 44).numpy(), 44, 200,
                         "./reonconstructed_eegs/E-generated-" + str(epoch))
        avg_cost_epoch = sum(costs_per_epoch) / len(costs_per_epoch)
        costs += [avg_cost_epoch]
        np.save("./reonconstructed_eegs/convVAE-lr1e-4-N4390-C44-L1004-E", np.asarray(costs))


if __name__ == "__main__":
    main()
