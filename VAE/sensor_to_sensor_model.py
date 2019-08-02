import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
import sys
sys.path.append("../data_loaders/")
sys.path.append("../ProgressiveGAN/EEG-GAN/")
from load_eegs_one_c_improved import EEGDataset
from eeggan.examples.conv_lin.model import Generator, Discriminator

class SensorToSensor(nn.Module):
    def __init__(self, num_channels=44, pretrained_encoder=None, pretrained_decoder=None):
        super(SensorToSensor, self).__init__()

        self.fc1 = nn.Sequential (
            nn.Linear(300, 250),

        )
        self.fc21 = nn.Linear(250, 200)
        self.fc22 = nn.Linear(250, 200)

        self.encoder = Discriminator(num_channels, n_out_linear=300)
        if pretrained_encoder != None:
            pretrained_model = torch.load(pretrained_encoder)

            # final weights and bias from GAN designed for 1d output
            pretrained_model.move_to_end("model.blocks.5.intermediate_sequence.3.weight_unscaled")
            pretrained_model.move_to_end("model.blocks.5.intermediate_sequence.3.bias") 
            pretrained_model.popitem(last=True)
            pretrained_model.popitem(last=True)

            self.encoder.load_state_dict(pretrained_model, strict=False)
        self.encoder.model.cur_block = 0 # there are six blocks and 5 is the last one
        self.encoder_block = self.encoder.model.cur_block
        self.encoder_model = self.encoder.model
        # modules = list(self.encoder.model.children())[0][:]
        # self.encoder = nn.Sequential(*modules)
        # print("encoder", self.encoder)
        self.decoder = Generator(num_channels, 200)
        if pretrained_decoder != None:
            self.decoder.load_state_dict(torch.load(pretrained_decoder))
        self.decoder.model.cur_block = 5
        # for param in self.decoder.parameters():
        #     param.requires_grad = False

    def transform_to_res(self, x):
        return self.encoder_model.downsample_to_block(
                    Variable(x[:, :, :, None].view(x.shape[0], 1, 768, 1), requires_grad=False),
                    self.encoder_block)
        
    def encode(self, x):
        x = self.transform_to_res(x)
        conv = self.encoder(x)
        h1 = self.fc1(conv)
        self.h1 = h1
        return self.fc21(h1), self.fc22(h1)

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
        # First Encode Brain Space Activity
        # input = torch.transpose(input, 1, 2)
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)

        # Then Decode to Sensor Space
        # out = self.decode(z)
        out = self.decode(z)

        return out, mu, logvar

    def generate(self, z):
        out = self.decode(z)
        return out

if __name__ == "__main__":
    model = SensorToSensor(num_channels=1)
    model = model.cuda()
    dataset = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=438, num_channels=44,
                       length=768, csv_file=None)
    s_t = dataset[0].cuda()
    # x_t = dataset.getSources(0)
    # x_t = np.random.rand(4, 7498, 44)
    # x_t = torch.from_numpy(x_t).type(torch.FloatTensor)
    out, mu, logvar = model(s_t)
    # print(out.shape)
