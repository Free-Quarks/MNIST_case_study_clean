import torch
from torch import nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h1_dim = 200, h2_dim = 60, z_dim = 10) -> None:
        super().__init__()
        # encoder matrices
        self.img_2_h1 = nn.Linear(input_dim, h1_dim)
        self.h1_2_h2 = nn.Linear(h1_dim, h2_dim)
        self.h2_2_mu = nn.Linear(h2_dim, z_dim)
        self.h2_2_logvar = nn.Linear(h2_dim, z_dim)
        
        # decoder matrices
        self.z_2_h2 = nn.Linear(z_dim, h2_dim)
        self.h2_2_h1 = nn.Linear(h2_dim, h1_dim)
        self.h1_2_img = nn.Linear(h1_dim, input_dim)

        # functional activations
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.img_2_h1(x))
        h2 = self.relu(self.h1_2_h2(h1))
        mu = self.h2_2_mu(h2) # only linear due to range of mu
        logvar = self.h2_2_logvar(h2) # only linear dur to range of logvar
        return mu, logvar
    
    def decode(self, z):
        h2 = self.relu(self.z_2_h2(z))
        h1 = self.relu(self.h2_2_h1(h2))
        x = torch.sigmoid(self.h1_2_img(h1)) # sigmoid for normalized pixel values
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        epsilon = torch.randn_like(mu) # input mu only provides the dimension for epsilon
        z_sampled = mu + torch.exp(0.5*logvar)*epsilon # reparameterization trick
        x_reconstructed = self.decode(z_sampled)
        return x_reconstructed, mu, logvar
    
def loss_function(x_reconstructed, x, mu, logvar):
    BCE = F.binary_cross_entropy(x_reconstructed, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
    
if __name__ == "__main__":
    x = torch.randn(4, 28*28) # 4 image batch of MNIST data
    vae = VariationalAutoEncoder(input_dim=28*28)
    x_reconstructed, mu, logvar = vae(x)
    print(x_reconstructed.shape)