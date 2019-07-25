import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
import sys
sys.path.append("../data_loaders/")
sys.path.append("../ProgressiveGAN/EEG-GAN/")
from forward_model_dataloader import ForwardModelDataset
from eeggan.examples.conv_lin.model import Generator


class ForwardLearned(nn.Module):
    def __init__(self, num_channels_out=44):
        super(ForwardLearned, self).__init__()

        def create_conv_sequence(in_filters, out_filters, kernel_size, stride, dilation=1, last=False):
            if last:
                final_activation = nn.Tanh()
            else:
                final_activation = nn.LeakyReLU(.2)

            return nn.Sequential(
                nn.Conv1d(in_filters, in_filters, kernel_size=kernel_size, stride=stride, dilation=dilation),
                nn.LeakyReLU(.2),
                nn.BatchNorm1d(in_filters),
                nn.Conv1d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_filters),
                final_activation,
            )
        self.encoder = nn.Sequential (
            create_conv_sequence(768, 384, 4, 2), # self.conv1 = 
            nn.MaxPool1d(4, stride=1), # self.pool1 = 
            create_conv_sequence(384, 192, 4, 2), # self.conv2 = 
            nn.MaxPool1d(4, stride=1), # self.pool2 = 
            create_conv_sequence(192, 96, 4, 1), # self.conv3 = 
            nn.MaxPool1d(3, stride=1), # self.pool3 = 
            create_conv_sequence(96, 1, 3, 1) # self.conv4 = 
        )

        self.fc1 = nn.Sequential (
            nn.Linear(451, 350),
            nn.Linear(350, 270)
        )
        self.fc21 = nn.Linear(270, 200)
        self.fc22 = nn.Linear(270, 200)

        self.decoder = Generator(num_channels_out, 200)
        self.decoder.model.cur_block = 5 # there are six blocks and 5 is the last one

    def encode(self, x):
        conv = self.encoder(x)
        h1 = self.fc1(conv.view(conv.shape[0], -1))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, x):
        transposed_conv = self.decoder(x)
        return transposed_conv

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
             eps = torch.FloatTensor(std.size()).normal_()
        esp = torch.randn(*mu.size())
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        # z = mu + std * esp
        return z

        # def reparametrize(self, mu, logvar):
        # std = logvar.mul(0.5).exp_()
        # if self.have_cuda:
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()
        # else:
        #     eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        # return eps.mul(std).add_(mu)

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, input):
        #things to add:  batch norm layers 

        # First Encode Brain Space Activity
        input = torch.transpose(input, 1, 2)
        mu, logvar = self.encode(input)

        z = self.reparameterize(mu, logvar)

        # Then Decode to Sensor Space
        out = self.decode(z)

        return out, mu, logvar


if __name__ == "__main__":
    model = ForwardLearned()
    # model = model.cuda
    # dataset = ForwardModelDataset(6, batch_size=4)
    # x_t = dataset.getSources(0)
    x_t = np.random.rand(4, 7498, 44)
    x_t = torch.from_numpy(x_t).type(torch.FloatTensor)
    out, mu, logvar = model(x_t)
    print(out.shape)
