import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comet_ml import Experiment
from vanilla_gan import VanillaGAN
import models


def load_data(batch_size=32):
    transform = ToTensor()
    mnist = MNIST('/home/vjjanush/mnist_data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)
    return mnist, loader


def update_checkpoint(epoch, global_step, prefix):
    with open(prefix + '/checkpoint', 'w') as f:
        json.dump({
            'restore_epoch': epoch,
            'global_step': global_step
        }, f, sort_keys=True, indent=4)


def read_checkpoint(prefix):
    try:
        with open(prefix + '/checkpoint', 'r') as f:
            d = json.load(f)
            return d['restore_epoch'], d['global_step']
    except:
        return 0, 0


def train(loader, gan):
    print_every = 100
    prefix = 'mnist/%s/%s' % (hyper_params['arch'], hyper_params['div'])
    os.makedirs(prefix, exist_ok=True)

    restore_epoch, global_step = read_checkpoint(prefix=prefix)
    global_step = 0

    if restore_epoch > 0:
        gan.load(
            generator_path = '%s/generator_%d.pt' % (prefix, restore_epoch),
            discriminator_path = '%s/discriminator_%d.pt' % (prefix, restore_epoch)
        )

    for epoch in range(hyper_params['n_epochs']):
        running_loss_D = 0.
        running_loss_G = 0.
        #scheduler.step()

        current_epoch = restore_epoch + epoch + 1

        for it, data in enumerate(loader):
            global_step += 1

            x = data[0].to(device).view(-1, 784)
            loss_D, loss_G = gan.update(x)

            running_loss_D += loss_D
            running_loss_G += loss_G

            experiment.log_metric('discriminator_loss', loss_D, step=global_step)
            experiment.log_metric('generator_loss', loss_G, step=global_step)

            if (it + 1) % print_every == 0:
                print("[epoch=%d, iteration=%4d] generator loss: %.4f, discriminator loss: %.4f" % (current_epoch, it + 1, running_loss_G / print_every, running_loss_D / print_every))
                running_loss_D = 0.
                running_loss_G = 0.

        gan.save(
            generator_path = '%s/generator_%d.pt' % (prefix, current_epoch), 
            discriminator_path = '%s/discriminator_%d.pt' % (prefix, current_epoch)
        )
        update_checkpoint(current_epoch, global_step, prefix=prefix)

        gan.train(False)
        fig = plot_generated_images(G, device=device)
        gan.train(True)
        fig.savefig('%s/generated_images_%d.png' % (prefix, current_epoch), format='png')
        experiment.log_figure(figure_name='Generated images [epoch=%d]' % current_epoch, figure=fig)


def plot_generated_images(G, ncols=5, nrows=5, device='cpu'):
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 15))
    images = G.generate(ncols * nrows, device=device).detach().cpu().numpy().reshape(ncols, nrows, 28, 28)
        
    for i in range(ncols):
        for j in range(nrows):
            axes[i, j].imshow(images[i, j], cmap='gray', aspect='auto')
            axes[i, j].axis('off')
            
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=100)
    parser.add_argument('--arch', type=str, choices=['fc', 'conv', 'conv_bn'], default='fc')
    parser.add_argument('--divergence', type=str, choices=['js', 'kl'], default='js')
    parser.add_argument('--generator_lr', type=float, default=1e-3)
    parser.add_argument('--discriminator_lr', type=float, default=1e-3)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    experiment = Experiment(api_key=os.environ["API_KEY"], project_name="gan-experiments")
    hyper_params = dict(
        generator_lr = args.generator_lr,
        discriminator_lr = args.discriminator_lr,
        batch_size = args.batch_size,
        n_epochs = args.n_epochs,
        latent_size = args.latent_size,
        arch = args.arch,
        div = args.divergence
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    with experiment.train():
        experiment.log_multiple_params(hyper_params)
    
        mnist, loader = load_data(hyper_params['batch_size'])

        if args.arch == 'fc':
            G = models.Generator(hyper_params['latent_size'])
            D = models.Discriminator()
        elif args.arch == 'conv':
            G = models.ConvGenerator(hyper_params['latent_size'], use_batch_norm=False)
            D = models.ConvDiscriminator(use_batch_norm=False)
        elif args.arch == 'conv_bn':
            G = models.ConvGenerator(hyper_params['latent_size'], use_batch_norm=True)
            D = models.ConvDiscriminator(use_batch_norm=True)

        #G.to(device)
        #D.to(device)
    
        optimizer_D = optim.Adam(D.parameters(), lr=hyper_params['discriminator_lr'])
        optimizer_G = optim.Adam(G.parameters(), lr=hyper_params['generator_lr'])

        gan = VanillaGAN(D, G, optimizer_D, optimizer_G, div=hyper_params['div'], device=device)
        print(gan)

        train(loader, gan)
