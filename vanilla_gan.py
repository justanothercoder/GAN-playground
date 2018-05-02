import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VanillaGAN(nn.Module):
    def __init__(self, D, G, optimizer_D, optimizer_G, div='js', device='cpu'):
        super(VanillaGAN, self).__init__()

        self.D = D
        self.G = G
        self.div = div

        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G

        self.device = device
        self.to(device)


    def forward(self, m):
        x = self.G.generate(m, device=self.device)
        return x.detach().to('cpu').numpy()


    def update(self, x):
        loss_D = self.update_D(x)
        loss_G = self.update_G(x.size(0))
        return loss_D, loss_G


    def update_D(self, x):
        self.optimizer_D.zero_grad()
        m = x.size(0)

        z = self.G.sample_prior(m).to(self.device)

        labels = torch.cat([
            torch.ones(m),
            torch.zeros(m)
        ]).to(self.device)

        outputs = torch.cat([
            self.D(x),
            self.D(self.G(z))
        ], dim=0)

        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        self.optimizer_D.step()

        return loss.item()


    def update_G(self, batch_size):
        self.optimizer_G.zero_grad()
        z = self.G.sample_prior(batch_size).to(self.device)

        if self.div == 'js':
            loss = -F.logsigmoid(self.D(self.G(z))).mean()
        elif self.div == 'kl':
            #loss = (F.logsigmoid(1 - D(G(z))) - F.logsigmoid(D(G(z)))).mean()
            loss = -self.D(self.G(z)).mean()

        loss.backward()
        self.optimizer_G.step()

        return loss.item()


    def load(self, generator_path, discriminator_path):
        self.G.load_state_dict(torch.load(generator_path))
        self.D.load_state_dict(torch.load(discriminator_path))


    def save(self, generator_path, discriminator_path):
        torch.save(self.G.state_dict(), generator_path)
        torch.save(self.D.state_dict(), discriminator_path)
