import numpy as np

import torch
from torch import nn
import sys
sys.path.append("../data_loaders/")
sys.path.append("../EEG-GAN/")
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
                nn.Conv1d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, dilation=dilation),
                final_activation,
            )

        self.conv1 = create_conv_sequence(44, 32, 5, 2)
        self.pool1 = nn.MaxPool1d(5, stride=2)
        self.conv2 = create_conv_sequence(32, 16, 5, 2)
        self.pool2 = nn.MaxPool1d(6, stride=1)
        self.conv3 = create_conv_sequence(16, 8, 6, 1)
        self.pool3 = nn.MaxPool1d(7, stride=1)
        self.conv4 = create_conv_sequence(8, 1, 6, 1)

        self.generator = Generator(num_channels_out, 200)
        self.generator.model.cur_block = 5 # there are six blocks and 5 is the last one

    def forward(self, input):

        # First Encode Brain Space Activity

        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        z = self.conv4(x)

        # Then Decode to Sensor Space
        out = self.generator(z)

        return out


if __name__ == "__main__":
    model = ForwardLearned()
    dataset = ForwardModelDataset(100)
    x_t = dataset.getSources(0)
    # x_t = np.random.rand(4, 44, 7498)
    # x_t = torch.from_numpy(x_t).type(torch.FloatTensor)
    z = model(x_t)
    print(z.shape)
