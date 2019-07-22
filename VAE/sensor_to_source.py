import torch
from torch import nn
import torchvision
import scipy


class InverseLearned(nn.Module):
    # input EEG
    def __init__(self, num_channels, num_dipoles, num_samples):
        super(InverseLearned, self).__init__()
        self.num_channels = num_channels
        self.num_dipoles = num_dipoles
        # 55 to 7498
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(num_samples, num_samples, kernel_size=5, stride=1, dilation=3),
            nn.BatchNorm1d(num_samples),
            nn.ReLU(num_samples)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(num_samples, num_samples, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_samples),
            nn.ReLU(num_samples)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(num_samples, num_samples, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_samples),
            nn.ReLU(num_samples)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(num_samples, num_samples, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_samples),
            nn.ReLU(num_samples)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose1d(num_samples, num_samples, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_samples),
            nn.ReLU(num_samples)
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose1d(num_samples, num_samples, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_samples),
            nn.ReLU(num_samples)
        )
        self.conv7 = nn.Sequential(
            nn.ConvTranspose1d(num_samples, num_samples, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_samples),
            nn.ReLU(num_samples)
        )
        self.conv8 = nn.Sequential(
            nn.ConvTranspose1d(num_samples, num_samples, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_samples),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x


if __name__ == "__main__":
    IL = InverseLearned(num_channels=45, num_dipoles=7498, num_samples=768)
    x = torch.rand((8, 768, 46))
    print(IL(x).shape)
