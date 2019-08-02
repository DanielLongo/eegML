import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
import sys
sys.path.append("../tsy935/RubinLab_neurotranslate_eeg-master/eeg/data/")
from data_loader import SeizureDataset
import torchvision.models as models

class Unflatten(nn.Module):
    def __init__(self):
        super(Unflatten, self).__init__()

    def forward(self, x):
        #converts 1d to 3
        # assert(((x.shape[1] / 10) ** .5 % 1) == 0), "Input not divisible by three and perfect square"
        # width = x.shape[1] // 3
        x = x.view(x.shape[0], 10, 10, 10)
        return x

class FeaturesToFeatures(nn.Module):
    def __init__(self, pretrained_encoder=True):
        super(FeaturesToFeatures, self).__init__()

        self.fc2_a = nn.Linear(1000, 1000) 
        self.fc2_b = nn.Linear(1000, 1000)

        self.encoder = models.squeezenet1_1(pretrained=pretrained_encoder)

        def create_gen_conv_block(c_in, c_out, kernel_size, stride, activation="ReLU"):
            if activation == "ReLU":
                return nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride),
                    nn.LeakyReLU(0.2)
                )
            if activation is None:
                return nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride),

                )

        self.decoder = nn.Sequential(
            nn.Linear(1000, 1000), # # 1083 so unflattenable
            Unflatten(),
            create_gen_conv_block(10, 8, 2, 1),
            create_gen_conv_block(8, 8, 3, 1),
            create_gen_conv_block(8, 8, 3, 1),
            create_gen_conv_block(8, 6, 3, 1),
            create_gen_conv_block(6, 6, 3, 1),
            create_gen_conv_block(6, 6, 3, 1),
            create_gen_conv_block(6, 4, 3, 1),
            create_gen_conv_block(4, 4, 3, 1),
            create_gen_conv_block(4, 3, 5, 2),
            create_gen_conv_block(3, 3, 6, 2),
            create_gen_conv_block(3, 3, 6, 2, activation=None),
            nn.ReLU()
        )
        
    def encode(self, x):
        conv = self.encoder(x)
        h1 = conv # possibly add hidden Linear
        self.h1 = h1
        return self.fc2_a(h1), self.fc2_b(h1)

    def decode(self, x):
        transposed_conv = self.decoder(x)
        return transposed_conv

    # def reparameterize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     if torch.cuda.is_available():
    #         eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     else:
    #          eps = torch.FloatTensor(std.size()).normal_()
    #     esp = torch.randn(*mu.size())
    #     eps = Variable(eps)
    #     z = eps.mul(std).add_(mu)
    #     # z = mu + std * esp
    #     return z
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar

    def generate(self, z):
        out = self.decode(z)
        return out

if __name__ == "__main__":
    model = FeaturesToFeatures()
    model = model.cuda()
    # dataset = SeizureDataset()
    x = torch.randn((4, 3, 224, 224)).cuda()
    # z, _ = model.encode(x)
    # x_hat = model.decode(z)
    # print(x_hat.shape)
    # print(model.encode(x)[0].shape)
    print(model.forward(x)[0].shape)
    # x_t = dataset.getSources(0)
    # x_t = np.random.rand(4, 7498, 44)
    # x_t = torch.from_numpy(x_t).type(torch.FloatTensor)
    # out, mu, logvar = model(s_t)
    # print(out.shape)
