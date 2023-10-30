import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define a function to flatten the input data
def flatten(x_train):
    """
    Flatten the input data.

    Args:
        x_train (torch.Tensor): The input dataset as a PyTorch tensor with shape (num_samples, width, height).

    Returns:
        Tuple[int, torch.Tensor]: A tuple containing the input dimension (width * height) and
        the flattened dataset as a PyTorch tensor.
    """
    dataset_shape = x_train.shape
    num_samples = dataset_shape[0]
    width, height = dataset_shape[1:]

    # Calculate input_dim
    input_dim = width * height

    # Reshape the dataset into a 2D format
    flattened_dataset = x_train.reshape(num_samples, input_dim)

    # Return both input_dim and flattened_dataset
    return width, height, input_dim, flattened_dataset

# Define the VAE architecture
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        encoder (nn.Sequential): The encoder neural network.
        decoder (nn.Sequential): The decoder neural network.
    """
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Two times latent_dim for mean and log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Sigmoid for output in [0, 1] range
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for the VAE.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            log_var (torch.Tensor): Log variance of the latent space.

        Returns:
            torch.Tensor: Sampled latent variable.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Decoded output, mean, and log variance of the latent space.
        """
        # Encoding
        enc_output = self.encoder(x)
        mu, log_var = enc_output.chunk(2, dim=-1)

        # Reparameterization
        z = self.reparameterize(mu, log_var)

        # Decoding
        dec_output = self.decoder(z)
        return dec_output, mu, log_var

    def vae_loss(self, recon_x, x, mu, log_var):
        """
        Calculate the VAE loss.

        Args:
            recon_x (torch.Tensor): Reconstructed data.
            x (torch.Tensor): Original input data.
            mu (torch.Tensor): Mean of the latent space.
            log_var (torch.Tensor): Log variance of the latent space.

        Returns:
            torch.Tensor: VAE loss.
        """
        reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kl_divergence
	

def train_vae(vae, train_loader, optimizer, num_epochs):
    """
    Train a Variational Autoencoder (VAE) model.

    Args:
        vae (VAE): The VAE model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        num_epochs (int): The number of training epochs.

    Returns:
        None
    """
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0]
            
            # Forward pass: Get reconstruction, mean, and log variance
            recon_x, mu, log_var = vae(x)
            
            # Calculate the VAE loss
            loss = vae.vae_loss(recon_x, x, mu, log_var)
            
            # Backpropagation
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print the average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')
