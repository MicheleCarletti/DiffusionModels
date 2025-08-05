"""
@author Michele Carletti
A simple example on denoising data in diffusion models 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Generate data
def generate_spiral(n_points=1000, noise_std=0.0):
    theta = np.sqrt(np.random.rand(n_points)) * 4 * np.pi
    r = theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.stack([x , y], axis=1)
    data += np.random.normal(0, noise_std, data.shape)  # Optional: add initial noise
    return torch.tensor(data, dtype=torch.float32)

# Diffusion forward process
def add_noise(x0, timesteps, beta_schedule):
    noise = torch.randn_like(x0)
    alphas = 1.0 - beta_schedule
    alphas_hat = torch.cumprod(alphas, dim=0)   # Quantity of original information preserved
    noisy_samples = []
    
    for t in range(timesteps):
        sqrt_alpha_hat = torch.sqrt(alphas_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alphas_hat[t])
        xt = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
        noisy_samples.append(xt.detach().numpy())
    
    return noisy_samples, noise

# Denoising model
class DenoiseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 + 1, 128),  # In: data coord + timestep
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 320),
            nn.ReLU(),
            nn.Linear(320, 2)   # Out: denoised coordinates
        )
    
    def forward(self, x, t):
        t = t.view(-1, 1).float() / 1000    # Timestep normalization
        x_in = torch.cat([x, t], dim=1)
        return self.mlp(x_in)

# Training
def train(model, data, alpha_hat, n_steps=1000, batch_size=128, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_hist = []

    for epoch in range(epochs):
        epoch_loss = 0
        perm = torch.randperm(len(data))
        
        for i in range(0, len(data), batch_size):   # For each batch
            idx = perm[i:i+batch_size]
            x0 = data[idx]
            t = torch.randint(0, n_steps, (x0.size(0), ))   # Choose random timesteps
           
            alpha_hat_t = alpha_hat[t]
            noise = torch.randn_like(x0)
            xt = torch.sqrt(alpha_hat_t.view(-1, 1)) * x0 + torch.sqrt(1 - alpha_hat_t.view(-1, 1)) * noise

            pred_noise = model(xt, t)
            loss = nn.MSELoss()(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        loss_hist.append(epoch_loss / (len(data) // batch_size))
        if epoch % 50 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
    
    return loss_hist

# Sampling (reverse process)
def sample(model, n_samples, n_steps, beta_schedule):
    model.eval()
    x = torch.randn(n_samples, 2)
    alphas = 1.0 - beta_schedule
    alpha_hat = torch.cumprod(alphas, dim=0)

    with torch.no_grad():
        for t in reversed(range(n_steps)):
            t_batch = torch.full((n_samples, ), t)
            z = torch.randn_like(x) if t > 0 else 0     # No noise is added in the last timestep
            beta_t = beta_schedule[t]
            alpha_t = alphas[t]
            alpha_hat_t = alpha_hat[t]

            pred_noise = model(x, t_batch)
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_hat_t)) * pred_noise) + torch.sqrt(beta_t) * z
    
    return x 

if __name__ == "__main__":

    # Step 1: generate dataset
    data = generate_spiral(n_points=10000)

    # Step 2: diffusion process
    n_steps = 100
    beta_schedule = torch.linspace(1e-4, 0.1, n_steps)     # Control the noise added 0.02 -> 2% added in every step
    noisy_samples, _ = add_noise(data, n_steps, beta_schedule)

    # Visualization
    for i in [0, 10, 30, 50, 70, 99]:
        plt.scatter(noisy_samples[i][:, 0], noisy_samples[i][:, 1], s=1)
        plt.title(f"Timestep {i}")
        plt.axis('equal')
        plt.show()
    
    # Step 3: model training
    model = DenoiseMLP()
    alpha_hat=torch.cumprod(1 - beta_schedule, dim=0)
    loss_final = train(model, data, alpha_hat, n_steps, epochs=1000)

    # Plot training loss
    plt.plot(loss_final)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()

    # Step 4: rversed sampling
    sampled = sample(model, n_samples=1000, n_steps=n_steps, beta_schedule=beta_schedule)

    # Final visualization
    plt.scatter(sampled[:, 0], sampled[:, 1], s=2, color='green', label='Generated')
    plt.scatter(data[:1000, 0], data[:1000, 1], s=2, color='red', label='Original')
    plt.legend()
    plt.axis('equal')
    plt.title("Generated Vs. Original")
    plt.show()