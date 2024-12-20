import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

NUM_GENRE = 21

class VAE(nn.Module):
    def __init__(self, hyperparams):
        super(VAE, self).__init__()
        if 'input_dim' not in hyperparams:
            raise ValueError('Input_dim should be defined')
        input_dim = hyperparams['input_dim']
        if 'hidden_dim' not in hyperparams:
            hidden = 128
        else:
            hidden = hyperparams['hidden_dim']
        input_dim = hyperparams['input_dim']
        if 'latent_dim' in hyperparams:
            latent_dim = hyperparams['latent_dim']
        else:
            latent_dim = 50
       # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden//2, latent_dim)
        self.logvar = nn.Linear(hidden//2, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Decode
        return self.decoder(z), mu, logvar
    
def loss_function4binary(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.BCELoss()(recon_x, x)
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def loss_function4regression(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

class GenreVAE(VAE):
    def __init__(self, hyperparams):
        super(VAE, self).__init__()
        if 'input_dim' not in hyperparams:
            raise ValueError('Input_dim should be defined')
        input_dim = hyperparams['input_dim']
        if 'hidden_dim' not in hyperparams:
            hidden = 128
        else:
            hidden = hyperparams['hidden_dim']
        input_dim = hyperparams['input_dim']
        if 'latent_dim' in hyperparams:
            latent_dim = hyperparams['latent_dim']
        else:
            latent_dim = 50
        if "embed_dim" in hyperparams:
            embed_dim = hyperparams["embed_dim"]
        else:
            embed_dim = 5
       # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        self.embeder = nn.Sequential(
            nn.Linear(NUM_GENRE, embed_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden//2 + embed_dim, latent_dim)
        self.logvar = nn.Linear(hidden//2 + embed_dim, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, ge):
        # Encode
        h = self.encoder(x)
        em = self.embeder(ge)
        h = torch.cat([h, em], dim=1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Decode
        return self.decoder(z), mu, logvar

class GenreVAE4GPU(VAE):
    def __init__(self, hyperparams):
        super(VAE, self).__init__()
        input_dim = hyperparams['input_dim']
        hidden_dim1 = 1024
        hidden_dim2 = 256
        hidden_dim3 = 64
        latent_dim = 50
        if "embed_dim" in hyperparams:
            embed_dim = hyperparams["embed_dim"]
        else:
            embed_dim = 5
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU()
        )
        self.embeder = nn.Sequential(
            nn.Linear(NUM_GENRE, embed_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim3 + embed_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim3 + embed_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, ge):
        # Encode
        h = self.encoder(x)
        em = self.embeder(ge)
        h = torch.cat([h, em], dim=1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Decode
        return self.decoder(z), mu, logvar