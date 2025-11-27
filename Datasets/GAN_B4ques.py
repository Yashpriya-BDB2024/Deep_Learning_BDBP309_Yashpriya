
# B4. Train a GAN model for a simulated dataset of random noise samples that resemble a 1D Gaussian distribution.
# The generator should learn to produce data samples that resemble real samples from a Gaussian distribution, and
# the discriminator will learn to distinguish between real and fake samples.
# You can use the following assumptions and hyperparameters in your implementation:
# The generator should have Linear – ReLU – Linear – ReLU – Linear layers.
# The discriminator should have Linear – LeakyReLU – Linear – LeakyReLU – Linear – Sigmoid.
# latent_dim = 10       # Dimension of the latent vector (input to the generator)
# hidden_dim = 128      # Hidden layer size
# output_dim = 1        # Output dimension (1D data point for simplicity)
# batch_size = 64
# num_epochs = 5000
# learning_rate = 0.0002
# Plot the real data and the generated data for visualization.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 10
hidden_dim = 128
output_dim = 1
batch_size = 64
num_epochs = 5000
learning_rate = 0.0002

def get_real_data(size):
    return torch.randn(size, output_dim)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def main():
    G = Generator()
    D = Discriminator()
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(D.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):    # Training
        # Train discriminator
        D.zero_grad()
        real_data = get_real_data(batch_size)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        output_real = D(real_data)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, latent_dim)
        fake_data = G(noise)
        output_fake = D(fake_data.detach())
        loss_fake = criterion(output_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Train generator
        G.zero_grad()
        output_fake_G = D(fake_data.detach())
        loss_G = criterion(output_fake_G, real_labels)
        loss_G.backward()
        optimizer_G.step()

        if (epoch+1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]; Loss_D: {loss_D.item():.4f}; Loss_G: {loss_G.item():.4f}")

    # Plot the real & generated data
    real_samples = get_real_data(1000).detach().numpy()
    fake_samples = G(torch.randn(1000, latent_dim)).detach().numpy()
    plt.hist(real_samples, bins=30, alpha=0.6, label="Real Gaussian Data")
    plt.hist(fake_samples, bins=30, alpha=0.6, label="Generated Data")
    plt.legend()
    plt.title("Real vs Generated 1D Gaussian samples")
    plt.show()

if __name__ == '__main__':
    main()
