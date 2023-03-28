"""
A categorical VAE that can train on Mario.
The notation very much follows the original VAE paper.
"""
from itertools import product
from typing import List

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
import torch.nn as nn
from time import time

import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader

from utils import load_data


class VAEMario(nn.Module):
    def __init__(
        self,
        w: int = 33,
        h: int = 16,
        z_dim: int = 4,
        n_sprites: int = 7,
        device: str = None,
    ):
        super(VAEMario, self).__init__()
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites
        self.z_dim = z_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
        ).to(self.device)

        self.p_z = Normal(
            torch.zeros(self.z_dim, device=self.device),
            torch.ones(self.z_dim, device=self.device),
        )

        self.train_data, self.test_data = load_data(device=self.device)


    def encode(self, x: torch.Tensor) -> Normal:
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> Categorical:
        logits = self.decoder(z)
        p_x_given_z = Categorical(
            logits=logits.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    def forward(self, x: torch.Tensor) -> List[Distribution]:
        q_z_given_x = self.encode(x.to(self.device))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: torch.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> torch.Tensor:
        x_ = x.to(self.device).argmax(dim=1)  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()


def fit(
    model: VAEMario,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
):
    model.train()
    running_loss = 0.0
    for (levels,) in data_loader:
        levels = levels.to(device)
        optimizer.zero_grad()
        q_z_given_x, p_x_given_z = model.forward(levels)
        loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(data_loader)


def test(
    model: VAEMario,
    test_loader: DataLoader,
    device: str,
    epoch: int = 0,
):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (levels,) in test_loader:
            levels.to(device)
            q_z_given_x, p_x_given_z = model.forward(levels)
            loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
            running_loss += loss.item()

    print(f"Epoch {epoch}. Loss in test: {running_loss / len(test_loader)}")
    return running_loss / len(test_loader)


def run(
    max_epochs: int = 2000,
    batch_size: int = 64,
    lr: int = 1e-3,
    save_every: int = None,
    overfit: bool = False,
):
    # Defining the name of the experiment
    timestamp = str(time()).replace(".", "")
    comment = f"{timestamp}_mariovae"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the data.
    training_tensors, test_tensors = load_data()

    # Creating datasets.
    dataset = TensorDataset(training_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Loading the model and optimizer
    print("Model:")
    vae = VAEMario()
    print(vae)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Training and testing.
    print(f"Training experiment {comment}")
    best_loss = np.Inf
    n_without_improvement = 0
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1} of {max_epochs}.")
        _ = fit(vae, optimizer, data_loader, device)
        test_loss = test(vae, test_loader, device, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            # Saving the best model so far.
            torch.save(vae.state_dict(), f"./models/{comment}_final.pt")
        else:
            if not overfit:
                n_without_improvement += 1

        if save_every is not None and epoch % save_every == 0 and epoch != 0:
            # Saving the model
            print(f"Saving the model at checkpoint {epoch}.")
            torch.save(vae.state_dict(), f"./models/{comment}_epoch_{epoch}.pt")

        # Early stopping:
        if n_without_improvement == 100:
            print("Stopping early")
            break


if __name__ == "__main__":
    run(overfit=True)
