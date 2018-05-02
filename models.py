import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x[:, 0]


class Generator(nn.Module):
    def __init__(self, latent_size=100):
        super(Generator, self).__init__()
        self.latent_size = latent_size

        self.fc1 = nn.Linear(latent_size, 300)
        self.fc2 = nn.Linear(300, 784)


    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z


    def sample_prior(self, n):
        return torch.randn(n, self.latent_size)


    def generate(self, batch_size, device='cpu'):
        z = self.sample_prior(batch_size).to(device)
        x = self.forward(z)
        return x


class ConvGenerator(nn.Module):
    def __init__(self, latent_size=100, use_batch_norm=True):
        super(ConvGenerator, self).__init__()
        self.latent_size = latent_size

        self.fc = nn.Linear(latent_size, 7 * 7 * 64)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batchnorm1 = nn.BatchNorm2d(64)
            self.batchnorm2 = nn.BatchNorm2d(32)
            self.batchnorm3 = nn.BatchNorm2d(16)


    def forward(self, z):
        if self.use_batch_norm:
            z = F.relu(self.batchnorm1(self.fc(z).view(-1, 64, 7, 7)))
            z = F.relu(self.batchnorm2(self.deconv1(z)))
            z = F.relu(self.batchnorm3(self.deconv2(z)))
        else:
            z = F.relu(self.fc(z).view(-1, 64, 7, 7))
            z = F.relu(self.deconv1(z))
            z = F.relu(self.deconv2(z))
        z = self.deconv3(z)
        return z.view(-1, 784)
    
    
    def sample_prior(self, n):
        return torch.randn(n, self.latent_size)


    def generate(self, batch_size, device='cpu'):
        z = self.sample_prior(batch_size).to(device)
        x = self.forward(z)
        return x


class ConvDiscriminator(nn.Module):
    def __init__(self, use_batch_norm=True):
        super(ConvDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=1)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batchnorm1 = nn.BatchNorm2d(16)
            self.batchnorm2 = nn.BatchNorm2d(32)
            self.batchnorm3 = nn.BatchNorm2d(64)
            self.batchnorm4 = nn.BatchNorm1d(100)
            self.batchnorm5 = nn.BatchNorm1d(100)


    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        if self.use_batch_norm:
            x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
            x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
            x = F.relu(self.batchnorm3(self.conv3(x)))
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 5 * 5)
        if self.use_batch_norm:
            x = F.relu(self.batchnorm4(self.fc1(x)))
            x = F.relu(self.batchnorm5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x[:, 0]
